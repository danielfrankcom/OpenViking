# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ICollection implementation on Valkey Search (HASH documents, dense vectors only).

Storage layout for a collection named ``C``:

* ``ovvec-meta:C`` - JSON with the collection meta and every index meta.
* ``ovvec:C:<id>`` - one HASH per record:
  ``__doc`` (the validated record as JSON, used for every read),
  the vector field as FLOAT32 bytes, ``__all`` = 1, and the derived values Valkey
  Search indexes (see ``filter.index_values``).
* FT index ``ovvec:C:<index name>`` over the ``ovvec:C:`` prefix.

Writes use plain commands, never MULTI or Lua: Valkey Search blocks the writing
client until the mutation is indexed, so a record is searchable as soon as the
write returns (read-after-write).

Two server limits shape the search code:

* ``search.max-vector-knn`` bounds K. A larger K is rejected by the server; the
  error is re-raised naming the setting and the K that was needed.
* ``search.max-nonvector-search-results-fetched`` ("the cap") bounds how many
  records every non-KNN query considers, per shard. The server truncates silently,
  so every non-KNN query is preceded or followed by a count, and a count that may
  have hit the cap raises instead of returning partial results.
"""

from __future__ import annotations

import copy
import json
import math
import random
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from openviking.storage.vectordb.collection.collection import ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
    UpsertDataResult,
)
from openviking.storage.vectordb.utils import validation
from openviking.storage.vectordb.utils.data_processor import DataProcessor
from openviking.storage.vectordb.utils.dict_utils import recursive_update_dict

from .filter import (
    ALL_FIELD,
    NUMERIC_TYPES,
    PATH_TYPE,
    TAG_MARK,
    TAG_SEPARATOR,
    TAG_TYPES,
    ancestors_field,
    clause_matches,
    client_side_clauses,
    compile_filter,
    depth_field,
    index_values,
    indexed_hash_fields,
    replace_client_side_clauses,
)

KNN_CAP_SETTING = "search.max-vector-knn"
NONVECTOR_CAP_SETTING = "search.max-nonvector-search-results-fetched"
# Valkey Search's default for NONVECTOR_CAP_SETTING.
DEFAULT_NONVECTOR_CAP = 100_000
# Valkey Search rejects a larger K whatever KNN_CAP_SETTING says.
MAX_KNN = 100_000

DOC_FIELD = "__doc"
DISTANCE_FIELD = "__distance"
BACKFILL_TIMEOUT_SECONDS = 300
SCAN_BATCH_SIZE = 1000
# Same amplification as the local engine's time-decay recall (index_manager_impl.cpp).
TIME_DECAY_RECALL_FACTOR = 3
TIME_DECAY_MAX_RECALL = 100_000
_COUNT_CONDITIONS = {
    "gt": lambda count, bound: count > bound,
    "gte": lambda count, bound: count >= bound,
    "lt": lambda count, bound: count < bound,
    "lte": lambda count, bound: count <= bound,
}


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _as_dict(flat_reply: List[Any]) -> Dict[str, Any]:
    """Turn a RESP2 ``[name, value, name, value, ...]`` reply into a dict keyed by name."""
    return {
        _text(name): value for name, value in zip(flat_reply[::2], flat_reply[1::2], strict=True)
    }


class ValkeyCollection(ICollection):
    def __init__(
        self,
        client: Any,
        cluster_mode: bool,
        name: str,
        meta: Dict[str, Any],
        nonvector_cap: int = DEFAULT_NONVECTOR_CAP,
    ):
        super().__init__()
        self._client = client
        self._cluster_mode = cluster_mode
        self._name = name
        self._meta = meta
        self._nonvector_cap = nonvector_cap
        self._load_schema()

    # ------------------------------------------------------------------
    # Lifecycle and metadata
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        client: Any,
        cluster_mode: bool,
        name: str,
        collection_meta: Dict[str, Any],
        nonvector_cap: int = DEFAULT_NONVECTOR_CAP,
    ):
        validation.validate_collection_meta_data(collection_meta)
        meta = {"collection": copy.deepcopy(collection_meta), "indexes": {}}
        collection = cls(client, cluster_mode, name, meta, nonvector_cap)
        if not collection._primary_key:
            raise ValueError("Valkey vector backend requires a primary key field")
        collection._save_meta()
        return collection

    @classmethod
    def open(
        cls,
        client: Any,
        cluster_mode: bool,
        name: str,
        nonvector_cap: int = DEFAULT_NONVECTOR_CAP,
    ) -> Optional["ValkeyCollection"]:
        raw = client.get(f"ovvec-meta:{name}")
        if raw is None:
            return None
        return cls(client, cluster_mode, name, json.loads(raw), nonvector_cap)

    def _load_schema(self) -> None:
        fields = self._meta["collection"].get("Fields", [])
        self._fields_dict = {field["FieldName"]: field for field in fields}
        self._primary_key = next((f["FieldName"] for f in fields if f.get("IsPrimaryKey")), "")
        self._vector_field = next(
            (f["FieldName"] for f in fields if f["FieldType"] == "vector"), ""
        )
        self._dimension = next((f.get("Dim", 0) for f in fields if f["FieldType"] == "vector"), 0)
        self._data_processor = DataProcessor(self._fields_dict, collection_name=self._name)

    def _save_meta(self) -> None:
        self._client.set(f"ovvec-meta:{self._name}", json.dumps(self._meta))

    def update(self, fields: Optional[Dict[str, Any]] = None, description: Optional[str] = None):
        changes: Dict[str, Any] = {}
        if fields is not None:
            changes["Fields"] = fields
        if description is not None:
            changes["Description"] = description
        if not changes:
            return
        validation.validate_collection_meta_data_for_update(changes)
        # Same merge as the local collection: new fields are appended to the schema.
        recursive_update_dict(self._meta["collection"], changes)
        self._load_schema()
        self._save_meta()

    def get_meta_data(self, raise_on_error: bool = False) -> Dict[str, Any]:
        return copy.deepcopy(self._meta["collection"])

    def close(self):
        # The adapter owns the client and closes it.
        pass

    def drop(self):
        for index_name in list(self._meta["indexes"]):
            self.drop_index(index_name)
        self.delete_all_data()
        self._client.delete(f"ovvec-meta:{self._name}")

    # ------------------------------------------------------------------
    # Indexes: one FT index per OpenViking index
    # ------------------------------------------------------------------

    def _ft_name(self, index_name: str) -> str:
        return f"ovvec:{self._name}:{index_name}"

    def _key(self, primary_key: Any) -> str:
        return f"ovvec:{self._name}:{primary_key}"

    def _indexed_types(self, index_name: str) -> Dict[str, str]:
        scalar_fields = self._meta["indexes"][index_name].get("ScalarIndex") or []
        return {field: self._fields_dict[field]["FieldType"] for field in scalar_fields}

    def _all_indexed_types(self) -> Dict[str, str]:
        types: Dict[str, str] = {}
        for index_name in self._meta["indexes"]:
            types.update(self._indexed_types(index_name))
        return types

    def create_index(self, index_name: str, meta_data: Dict[str, Any]):
        index_meta = copy.deepcopy(meta_data)
        index_meta["IndexName"] = index_name
        self._create_ft_index(index_name, index_meta)
        self._meta["indexes"][index_name] = index_meta
        self._save_meta()

    def _create_ft_index(self, index_name: str, index_meta: Dict[str, Any]) -> None:
        vector_index = index_meta.get("VectorIndex", {})
        vector_args = [
            "TYPE",
            "FLOAT32",
            "DIM",
            str(self._dimension),
            "DISTANCE_METRIC",
            str(vector_index.get("Distance", "cosine")).upper(),
        ]
        for name, value in (vector_index.get("Params") or {}).items():
            vector_args += [name, str(value)]
        schema = [
            ALL_FIELD,
            "NUMERIC",
            self._vector_field,
            "VECTOR",
            str(vector_index.get("IndexType", "flat")).upper(),
            str(len(vector_args)),
            *vector_args,
        ]
        expected_attributes = {ALL_FIELD, self._vector_field}
        for field in index_meta.get("ScalarIndex") or []:
            if field not in self._fields_dict:
                raise ValueError(f"ScalarIndex field {field!r} is not in the collection schema")
            field_type = self._fields_dict[field]["FieldType"]
            schema += self._schema_args(field, field_type)
            expected_attributes.update(self._attribute_names(field, field_type))
        # redis is an optional dependency; this module is imported even without it.
        from redis.exceptions import ResponseError

        try:
            self._client.execute_command(
                "FT.CREATE",
                self._ft_name(index_name),
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._key(""),
                "SCHEMA",
                *schema,
            )
        except ResponseError as error:
            # FT.CREATE is not idempotent, and two callers can legitimately send it for
            # the same index: redis-py resends a command that hit the client-side socket
            # timeout while the server was still creating the index, and several
            # OpenViking processes can start against an empty deployment at once. Accept
            # an existing index only if it has this collection's prefix and attributes.
            if "already exists" not in str(error):
                raise
            if not self._ft_index_matches(index_name, expected_attributes):
                raise
        self._wait_for_backfill(index_name)

    def _ft_index_matches(self, index_name: str, expected_attributes: Set[str]) -> bool:
        # Compares the key prefix and the attribute names only, not the attribute types
        # or the vector dimension and metric: enough to tell this collection's index
        # from another one, which is the case this guards against.
        info = _as_dict(self._client.execute_command("FT.INFO", self._ft_name(index_name)))
        definition = _as_dict(info.get("index_definition") or [])
        prefixes = [_text(prefix) for prefix in definition.get("prefixes") or []]
        identifiers = {
            _text(_as_dict(attribute).get("identifier", ""))
            for attribute in info.get("attributes") or []
        }
        return prefixes == [self._key("")] and identifiers == expected_attributes

    @staticmethod
    def _schema_args(field: str, field_type: str) -> List[str]:
        tag = ["TAG", "SEPARATOR", TAG_SEPARATOR, "CASESENSITIVE"]
        if field_type in NUMERIC_TYPES:
            return [field, "NUMERIC"]
        if field_type in TAG_TYPES:
            return [field, *tag]
        if field_type == PATH_TYPE:
            return [field, *tag, ancestors_field(field), *tag, depth_field(field), "NUMERIC"]
        raise NotImplementedError(
            f"Valkey vector backend cannot index {field_type} field {field!r}"
        )

    @staticmethod
    def _attribute_names(field: str, field_type: str) -> List[str]:
        """The FT attribute names that ``_schema_args`` creates for ``field``."""
        if field_type == PATH_TYPE:
            return [field, ancestors_field(field), depth_field(field)]
        return [field]

    def _wait_for_backfill(self, index_name: str) -> None:
        # An index created over existing keys is filled by a background scan; wait so
        # that queries right after create/update see every record. In cluster mode the
        # default FT.INFO scope is the local node only; CLUSTER covers every shard.
        scope = ["CLUSTER"] if self._cluster_mode else []
        deadline = time.monotonic() + BACKFILL_TIMEOUT_SECONDS
        while True:
            info = _as_dict(
                self._client.execute_command("FT.INFO", self._ft_name(index_name), *scope)
            )
            if _text(info.get("backfill_in_progress", "0")) == "0":
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"Valkey Search backfill of {index_name!r} did not finish")
            time.sleep(0.05)

    def has_index(self, index_name: str) -> bool:
        return index_name in self._meta["indexes"]

    def get_index(self, index_name: str) -> Any:
        # Like the http collection, return the index meta: the FT index has no
        # client-side object, and no caller uses IIndex methods on the result.
        return self.get_index_meta_data(index_name) or None

    def get_index_meta_data(self, index_name: str) -> Dict[str, Any]:
        return copy.deepcopy(self._meta["indexes"].get(index_name, {}))

    def list_indexes(self) -> List[str]:
        return list(self._meta["indexes"])

    def drop_index(self, index_name: str):
        if index_name not in self._meta["indexes"]:
            return
        self._client.execute_command("FT.DROPINDEX", self._ft_name(index_name))
        del self._meta["indexes"][index_name]
        self._save_meta()

    def update_index(
        self,
        index_name: str,
        scalar_index: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        index_meta = self._meta["indexes"][index_name]
        if description is not None:
            index_meta["Description"] = description
        if scalar_index is not None and scalar_index != index_meta.get("ScalarIndex"):
            # Valkey Search has no FT.ALTER: recreate the FT index. Records already
            # carry every derived value they need only for the old field list, so
            # rewrite them before the new index backfills.
            index_meta["ScalarIndex"] = list(scalar_index)
            self._client.execute_command("FT.DROPINDEX", self._ft_name(index_name))
            self._rewrite_all_records()
            self._create_ft_index(index_name, index_meta)
        self._save_meta()

    def _rewrite_all_records(self) -> None:
        # Rewriting with ttl=0 clears per-record TTLs; no OpenViking caller combines
        # TTL writes with ScalarIndex changes.
        for keys in self._scan_record_keys():
            ids = [_text(key)[len(self._key("")) :] for key in keys]
            self._write_records(list(self._read_records(ids).values()), ttl=0)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def upsert_data(self, data_list: List[Dict[str, Any]], ttl=0):
        records = [self._data_processor.validate_and_process(item) for item in data_list]
        return self._write_records(records, ttl)

    def _write_records(self, records: List[Dict[str, Any]], ttl: int) -> UpsertDataResult:
        indexed_types = self._all_indexed_types()
        derived_fields = indexed_hash_fields(indexed_types)
        pipeline = self._client.pipeline(transaction=False)
        for record in records:
            key = self._key(record[self._primary_key])
            doc = {name: value for name, value in record.items() if name != self._vector_field}
            vector = record.get(self._vector_field) or []
            for_index = self._data_processor.convert_fields_dict_for_index(doc)
            indexed = index_values(for_index, indexed_types)
            hash_fields = {
                DOC_FIELD: json.dumps(doc, ensure_ascii=False),
                self._vector_field: struct.pack(f"<{len(vector)}f", *vector),
                ALL_FIELD: "1",
                **indexed,
            }
            pipeline.hset(key, mapping=hash_fields)
            # HSET overwrites in place, so a concurrent search never misses the
            # record; derived values the new record no longer has are removed after.
            stale_fields = sorted(derived_fields - set(indexed))
            if stale_fields:
                pipeline.hdel(key, *stale_fields)
            if ttl > 0:
                pipeline.pexpire(key, int(ttl * 1000))
            else:
                pipeline.persist(key)
        if records:
            pipeline.execute()
        return UpsertDataResult(ids=[record[self._primary_key] for record in records])

    def update_data(self, data_list: List[Dict[str, Any]]):
        if not data_list:
            return UpsertDataResult()
        primary_keys = []
        for item in data_list:
            if self._primary_key not in item:
                raise ValueError(f"primary key '{self._primary_key}' is required for update")
            primary_keys.append(item[self._primary_key])
        existing = self._read_records(primary_keys)
        missing = [primary_key for primary_key in primary_keys if primary_key not in existing]
        if missing:
            # The storage layer maps this "not found" message to NOT_FOUND.
            raise ValueError(f"record not found for primary key(s): {missing}")
        merged = [{**existing[item[self._primary_key]], **item} for item in data_list]
        return self.upsert_data(merged)

    def fetch_data(self, primary_keys: List[Any]) -> FetchDataInCollectionResult:
        found = self._read_records(primary_keys)
        return FetchDataInCollectionResult(
            items=[DataItem(id=key, fields=found[key]) for key in primary_keys if key in found],
            ids_not_exist=[key for key in primary_keys if key not in found],
        )

    def _read_records(self, primary_keys: List[Any]) -> Dict[Any, Dict[str, Any]]:
        """Return the stored records that exist, keyed by primary key."""
        if not primary_keys:
            return {}
        pipeline = self._client.pipeline(transaction=False)
        for primary_key in primary_keys:
            pipeline.hmget(self._key(primary_key), [DOC_FIELD, self._vector_field])
        replies = pipeline.execute()
        return {
            primary_key: self._decode(doc, vector)
            for primary_key, (doc, vector) in zip(primary_keys, replies, strict=True)
            if doc is not None
        }

    def _decode(self, doc: Any, vector: Optional[bytes]) -> Dict[str, Any]:
        fields = json.loads(doc)
        if vector is not None:
            fields[self._vector_field] = list(struct.unpack(f"<{len(vector) // 4}f", vector))
        return fields

    def delete_data(self, primary_keys: List[Any]):
        if primary_keys:
            self._client.delete(*[self._key(primary_key) for primary_key in primary_keys])

    def delete_all_data(self):
        for keys in self._scan_record_keys():
            self._client.delete(*keys)

    def _scan_record_keys(self):
        """Yield batches of this collection's record keys (works with no FT index)."""
        pattern = "".join("\\" + ch if ch in "*?[]\\" else ch for ch in self._key("")) + "*"
        batch = []
        # scan_iter covers every primary in cluster mode.
        for key in self._client.scan_iter(match=pattern, count=SCAN_BATCH_SIZE):
            batch.append(key)
            if len(batch) == SCAN_BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _resolve_filter(
        self, index_name: str, filters: Optional[Dict[str, Any]]
    ) -> Tuple[str, Optional[Set[str]], int]:
        """Compile ``filters`` into a query; evaluate contains/regex clauses here.

        Returns ``(query, exact_ids, candidates)``. For a filter without contains or
        regex clauses the query is exact, ``exact_ids`` is None and ``candidates`` 0.

        Valkey Search cannot match substrings or regexes, so otherwise the query is a
        superset that takes every such clause as true (a superset because the DSL
        negates only single clauses, never subtrees). ``exact_ids`` are the ids that
        match the whole filter and ``candidates`` is the number of records the
        superset matches; callers rank the superset and keep only ``exact_ids``.

        Each candidate's clauses are evaluated on its stored record, as the local
        engine evaluates them on its index keys (``filter.clause_matches``). The rest
        of the filter stays with Valkey Search: a candidate matches iff the filter, with
        the clauses fixed to that candidate's results, matches it. One query per
        distinct combination of results decides every candidate with that
        combination, so a single clause needs at most one extra query.

        Cost: every candidate's document is read, and every query here is non-vector,
        so the superset must stay under the non-vector cap (``_count`` checks it).
        """
        converted = self._data_processor.convert_filter_for_index(copy.deepcopy(filters or {}))
        field_types = self._indexed_types(index_name)
        clauses = client_side_clauses(converted)
        if not clauses:
            return compile_filter(converted, field_types), None, 0

        def query_with(results: List[bool]) -> str:
            return compile_filter(replace_client_side_clauses(converted, results), field_types)

        superset = query_with([True] * len(clauses))
        docs = self._matching_docs(index_name, superset)
        ids_by_results: Dict[Tuple[bool, ...], List[str]] = {}
        for record_id, doc in docs.items():
            record = self._data_processor.convert_fields_dict_for_index(json.loads(doc))
            results = tuple(
                clause_matches(clause, record, field_types.get(clause.get("field", "")))
                for clause in clauses
            )
            ids_by_results.setdefault(results, []).append(record_id)
        exact_ids: Set[str] = set()
        for results, ids in ids_by_results.items():
            if all(results):
                # With every clause true the filter is the superset query itself.
                exact_ids.update(ids)
            else:
                exact_ids.update(
                    self._matching_ids(index_name, query_with(list(results))) & set(ids)
                )
        return superset, exact_ids, len(docs)

    def _count(self, index_name: str, query: str) -> int:
        """Count the records a query matches; raise if the non-vector cap may have cut it."""
        reply = self._client.execute_command(
            "FT.SEARCH", self._ft_name(index_name), query, "LIMIT", "0", "0"
        )
        total = int(reply[0])
        if total < self._nonvector_cap:
            return total
        # The cap applies per shard: in cluster mode a larger total is complete when
        # no shard holds that many records.
        if self._cluster_mode and self._largest_shard_size(index_name) < self._nonvector_cap:
            return total
        raise RuntimeError(
            f"Valkey Search may have returned partial results: the query matches at least "
            f"{self._nonvector_cap} records on one shard, the most a non-vector query "
            f"considers ({NONVECTOR_CAP_SETTING}). Raise {NONVECTOR_CAP_SETTING} above the "
            f"number of records per shard (4294967295 removes the cap)."
        )

    def _largest_shard_size(self, index_name: str) -> int:
        # Without the CLUSTER scope, FT.INFO reports the node's own records.
        replies = self._client.execute_command(
            "FT.INFO", self._ft_name(index_name), target_nodes=self._client.PRIMARIES
        )
        if not isinstance(replies, dict):
            replies = {"only primary": replies}
        return max(int(_text(_as_dict(reply)["num_docs"])) for reply in replies.values())

    def _matching_docs(self, index_name: str, query: str) -> Dict[str, Any]:
        """Return ``{id: stored __doc}`` for every record a non-vector query matches."""
        total = self._count(index_name, query)
        if total == 0:
            return {}
        reply = self._client.execute_command(
            "FT.SEARCH",
            self._ft_name(index_name),
            query,
            "RETURN",
            "1",
            DOC_FIELD,
            "LIMIT",
            "0",
            str(total),
        )
        return {record_id: values[DOC_FIELD] for record_id, values in self._rows(reply)}

    def _matching_ids(self, index_name: str, query: str) -> Set[str]:
        """Return the id of every record a non-vector query matches."""
        total = self._count(index_name, query)
        if total == 0:
            return set()
        reply = self._client.execute_command(
            "FT.SEARCH", self._ft_name(index_name), query, "NOCONTENT", "LIMIT", "0", str(total)
        )
        prefix_length = len(self._key(""))
        return {_text(key)[prefix_length:] for key in reply[1:]}

    def _return_fields(self, output_fields: Optional[List[str]], *extra: str) -> List[str]:
        fields = [DOC_FIELD, *extra]
        if output_fields is None or self._vector_field in output_fields:
            fields.append(self._vector_field)
        return ["RETURN", str(len(fields)), *fields]

    def _project(self, fields: Dict[str, Any], output_fields: Optional[List[str]]):
        # Same projection as the local collection: every requested field, None if absent.
        names = output_fields or list(self._fields_dict)
        return {name: fields.get(name) for name in names}

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: Optional[List[float]] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
        output_fields: Optional[List[str]] = None,
        advance: Optional[Dict[str, Any]] = None,
        return_detail_info: bool = False,
    ) -> SearchResult:
        # sparse_vector is ignored: collections are dense-only (sparse_weight > 0 is
        # rejected at config time), but hybrid embedders still send sparse vectors.
        if not dense_vector:
            return SearchResult()
        time_decay = (advance or {}).get("time_decay")
        window = limit + offset
        if time_decay is not None:
            window = min(window * TIME_DECAY_RECALL_FACTOR, TIME_DECAY_MAX_RECALL)
        query_filter, exact_ids, candidates = self._resolve_filter(index_name, filters)
        k, first = window, (0 if time_decay is not None else offset)
        if exact_ids is not None:
            # The filter query is a superset: rank all of it, keep exact matches below.
            if candidates == 0:
                return SearchResult()
            k, first = candidates, 0
        prefilter = f"({query_filter})" if filters else "*"
        query = f"{prefilter}=>[KNN {k} @{self._vector_field} $vector AS {DISTANCE_FIELD}]"
        try:
            reply = self._client.execute_command(
                "FT.SEARCH",
                self._ft_name(index_name),
                query,
                "PARAMS",
                "2",
                "vector",
                struct.pack(f"<{len(dense_vector)}f", *dense_vector),
                *self._return_fields(output_fields, DISTANCE_FIELD),
                "LIMIT",
                str(first),
                str(k - first),
            )
        except Exception as error:
            # OpenViking's fetch-all callers ask for K = 100000 (filter(limit=100000)),
            # and Valkey Search rejects K above the setting (default 10000).
            if "KNN parameter" not in str(error) or k < 1:
                raise
            needed = f"at least {k}" if k <= MAX_KNN else f"{k}, above its maximum of {MAX_KNN}"
            raise RuntimeError(
                f"Valkey Search rejected a vector query with K={k}: {KNN_CAP_SETTING} "
                f"must be {needed}. Server error: {error}"
            ) from error
        metric = str(self._meta["indexes"][index_name]["VectorIndex"].get("Distance", "cosine"))
        hits = []
        for primary_key, values in self._rows(reply):
            record = self._decode(values[DOC_FIELD], values.get(self._vector_field))
            score = _score_from_distance(metric.lower(), float(values[DISTANCE_FIELD]))
            hits.append((primary_key, record, score))
        if exact_ids is not None:
            hits = [hit for hit in hits if hit[0] in exact_ids][:window]
            if time_decay is None:
                hits = hits[offset:]
        if time_decay is not None:
            items = self._apply_time_decay(hits, time_decay, output_fields)
            return SearchResult(data=items[offset : offset + limit])
        return SearchResult(
            data=[
                SearchItemResult(id=key, fields=self._project(record, output_fields), score=score)
                for key, record, score in hits
            ]
        )

    def _apply_time_decay(
        self,
        hits: List[Tuple[Any, Dict[str, Any], float]],
        decay: Dict[str, Any],
        output_fields: Optional[List[str]],
    ) -> List[SearchItemResult]:
        """Re-rank like the local engine: score * exp(rate * max(0, |origin - t| - offset))."""
        rate = math.log(decay["decay"]) / decay["scale_ms"]
        ranked = []
        for primary_key, record, score in hits:
            time_score = None
            updated = record.get(decay["field"])
            if updated:
                updated_ms = self._data_processor.parse_datetime_to_epoch_ms(updated)
                age_beyond_protection = max(
                    0.0, abs(decay["origin_ms"] - updated_ms) - decay["offset_ms"]
                )
                time_score = math.exp(rate * age_beyond_protection)
            decayed_score = score if time_score is None else score * time_score
            item = SearchItemResult(
                id=primary_key,
                fields=self._project(record, output_fields),
                score=decayed_score,
                origin_score=score,
                addition_score=time_score,
            )
            ranked.append((decayed_score, item))
        # sort() is stable, matching the local engine's stable_sort on equal scores.
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked]

    def _rows(self, reply: List[Any]):
        """Yield (primary key, returned fields) of an FT.SEARCH reply in server order."""
        # The reply is [total, key, [field, value, ...], key, [field, value, ...], ...].
        prefix_length = len(self._key(""))
        for key, values in zip(reply[1::2], reply[2::2], strict=True):
            yield _text(key)[prefix_length:], _as_dict(values)

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: Optional[str] = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        if self._indexed_types(index_name).get(field) not in NUMERIC_TYPES:
            # The local engine's sorter returns nothing for path/string fields either.
            raise NotImplementedError(
                f"Valkey vector backend sorts only by numeric fields, not {field!r}"
            )
        direction = "DESC" if (order or "desc").lower() == "desc" else "ASC"
        # Offset pages (scroll) must see every record exactly once, so the order must be
        # total and must keep records that lack the field. FT.SEARCH SORTBY breaks ties
        # arbitrarily per query; FT.AGGREGATE sorts by (has field, field, key) instead,
        # which puts records without the field last.
        # Open decision (missing sort field): records without the field are kept and
        # sorted last; the local sorter drops them without a filter and places them
        # arbitrarily with one.
        query, exact_ids, candidates = self._resolve_filter(index_name, filters)
        first, count = offset, limit
        if exact_ids is None:
            # FT.AGGREGATE returns no total, so count separately to detect the cap.
            self._count(index_name, query)
        else:
            # The query is a superset whose size _resolve_filter has counted (and
            # checked against the cap): sort all of it, keep exact matches below.
            if candidates == 0:
                return SearchResult()
            first, count = 0, candidates
        reply = self._client.execute_command(
            "FT.AGGREGATE",
            self._ft_name(index_name),
            query,
            "LOAD",
            "2",
            "@__key",
            f"@{field}",
            "APPLY",
            f"exists(@{field})",
            "AS",
            "__has",
            "SORTBY",
            "6",
            "@__has",
            "DESC",
            f"@{field}",
            direction,
            "@__key",
            direction,
            "MAX",
            str(first + count),
            "LIMIT",
            str(first),
            str(count),
        )
        # FT.AGGREGATE before 1.3 can LOAD only indexed attributes, so read the
        # documents by key afterwards.
        # The reply is [total, [name, value, ...], ...], one list per row.
        prefix_length = len(self._key(""))
        ids = [_text(_as_dict(row)["__key"])[prefix_length:] for row in reply[1:]]
        if exact_ids is not None:
            ids = [record_id for record_id in ids if record_id in exact_ids][
                offset : offset + limit
            ]
        found = self._read_records(ids)
        # Like the local collection, the sort value is reported as the score.
        return SearchResult(
            data=[
                SearchItemResult(
                    id=record_id,
                    fields=self._project(found[record_id], output_fields),
                    score=found[record_id].get(field),
                )
                for record_id in ids
                if record_id in found
            ]
        )

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        advance: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        random_vector = [random.uniform(-1, 1) for _ in range(self._dimension)]
        return self.search_by_vector(
            index_name, random_vector, limit, offset, filters, None, output_fields
        )

    def search_by_id(
        self,
        index_name: str,
        id: Any,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        found = self._read_records([id])
        if id not in found:
            return SearchResult()
        vector = found[id].get(self._vector_field)
        return self.search_by_vector(
            index_name, vector, limit, offset, filters, None, output_fields
        )

    def search_by_keywords(self, index_name: str, *args, **kwargs) -> SearchResult:
        # Same as the local backend: no server-side text vectorization or BM25 over
        # "content" (USE_CONTENT_FIELD is False, so content is never written).
        raise NotImplementedError("Valkey vector backend does not support keyword search")

    def search_by_multimodal(self, index_name: str, *args, **kwargs) -> SearchResult:
        raise NotImplementedError("Valkey vector backend does not vectorize multimodal input")

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        cond: Optional[Dict[str, Any]] = None,
    ) -> AggregateResult:
        if op != "count":
            raise NotImplementedError(f"Valkey vector backend does not support aggregate op {op!r}")
        query, exact_ids, _ = self._resolve_filter(index_name, filters)
        if field is None:
            total = len(exact_ids) if exact_ids is not None else self._count(index_name, query)
            return AggregateResult(agg={"_total": total}, op=op)
        if exact_ids is not None:
            # No OpenViking caller groups counts (adapters call aggregate_data without
            # a field), so contains/regex are not combined with GROUPBY.
            raise NotImplementedError(
                "Valkey vector backend does not support contains/regex in a grouped count"
            )
        field_type = self._indexed_types(index_name).get(field)
        if field_type not in NUMERIC_TYPES and field_type not in {"string", "text"}:
            raise NotImplementedError(f"Valkey vector backend cannot group by {field_type} field")
        # FT.AGGREGATE returns no total, so count separately to detect the cap.
        self._count(index_name, query)
        reply = self._client.execute_command(
            "FT.AGGREGATE",
            self._ft_name(index_name),
            query,
            "LOAD",
            "1",
            f"@{field}",
            "GROUPBY",
            "1",
            f"@{field}",
            "REDUCE",
            "COUNT",
            "0",
            "AS",
            "__count",
        )
        counts: Dict[str, int] = {}
        for row in map(_as_dict, reply[1:]):
            value = _text(row.get(field, "")).strip(TAG_MARK)
            counts[value] = int(_text(row["__count"]))
        for name, bound in (cond or {}).items():
            counts = {k: v for k, v in counts.items() if _COUNT_CONDITIONS[name](v, bound)}
        return AggregateResult(agg=counts, op=op, field=field)


def _score_from_distance(metric: str, distance: float) -> float:
    """Convert a Valkey Search KNN distance to the local engine's score scale.

    Valkey reports COSINE as 1 - cos, IP as 1 - ip and L2 as the squared distance.
    The local engine scores cosine as (1 + cos) / 2 clamped to [0, 1], IP as the raw
    inner product and L2 as 1 - squared distance, so thresholds mean the same thing.
    """
    if metric == "cosine":
        return min(1.0, max(0.0, 1.0 - distance / 2.0))
    return 1.0 - distance
