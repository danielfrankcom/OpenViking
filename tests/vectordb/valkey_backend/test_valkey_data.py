# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Record operations, read-after-write, scroll/fetch-all and sparse handling."""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import Eq, In, PathScope
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking_cli.session.user_id import UserIdentifier

pytestmark = pytest.mark.integration


def _create(adapter, dimension=4):
    name = adapter.collection_name
    adapter.create_collection(
        name,
        CollectionSchemas.context_collection(name, dimension),
        distance="cosine",
        sparse_weight=0.0,
        index_name="default",
    )
    return adapter


def _record(i, **extra):
    record = {
        "id": f"r{i:05d}",
        "uri": f"viking://resources/d{i % 10}/f{i}.md",
        "account_id": "acct",
        "level": 2,
        "updated_at": "2026-01-01T00:00:00Z",
        "vector": [1.0, (i % 13) / 13.0, (i % 7) / 7.0, 0.1],
    }
    record.update(extra)
    return record


def test_upsert_get_delete(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    assert adapter.upsert([_record(1), _record(2)]) == ["r00001", "r00002"]
    assert [r["id"] for r in adapter.get(["r00002", "missing", "r00001"])] == ["r00002", "r00001"]
    assert adapter.delete(ids=["r00001", "missing"]) == 2
    assert [r["id"] for r in adapter.get(["r00001", "r00002"])] == ["r00002"]
    assert adapter.count() == 1


def test_upsert_replaces_whole_record_and_derived_index_values(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    adapter.upsert(_record(1, search_tags=["a=1", "b=2"], acl_direct_grants=["read:user:x"]))
    assert adapter.count(In("search_tags", ["a=1"])) == 1
    adapter.upsert(_record(1, uri="viking://resources/moved.md"))
    assert adapter.count(In("search_tags", ["a=1"])) == 0
    assert adapter.count(In("acl_direct_grants", ["read:user:x"])) == 0
    assert adapter.count(PathScope("uri", "viking://resources/d1", depth=-1)) == 0
    assert adapter.get(["r00001"])[0]["search_tags"] == []


def test_update_data_merges_and_reports_missing(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    adapter.upsert(_record(1, name="before", search_tags=["k=v"]))
    assert adapter.update_data(
        [{"id": "r00001", "name": "after", "uri": "viking://resources/new"}]
    ) == ["r00001"]
    record = adapter.get(["r00001"])[0]
    assert (record["name"], record["uri"], record["search_tags"]) == (
        "after",
        "viking://resources/new",
        ["k=v"],
    )
    assert record["vector"] == pytest.approx(_record(1)["vector"])
    assert adapter.count(Eq("uri", "viking://resources/new")) == 1
    with pytest.raises(ValueError, match="not found"):
        adapter.update_data([{"id": "missing", "name": "x"}])


async def test_backend_update_maps_missing_record_to_not_found(make_valkey_adapter, valkey_config):
    name = make_valkey_adapter().collection_name
    backend = VikingVectorIndexBackend(valkey_config(name, 4))
    settings = SimpleNamespace(vectordb=valkey_config(name, 4), dedicated_vectordb=False)
    backend.set_vector_config_resolver(SimpleNamespace(resolve=AsyncMock(return_value=settings)))
    ctx = RequestContext(user=UserIdentifier("acct", "alice"), role=Role.ROOT)
    try:
        assert await backend.create_collection(name, CollectionSchemas.context_collection(name, 4))
        result = await backend.update({"id": "missing", "name": "x"}, ctx=ctx)
        assert (result.ok, result.error_code) == (False, "NOT_FOUND")
        await backend.upsert(_record(1), ctx=ctx)
        result = await backend.update({"id": "r00001", "name": "x"}, ctx=ctx)
        assert result.ok and result.ids == ["r00001"]
    finally:
        await backend.close()


def test_ttl_expires_and_plain_upsert_clears_it(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    collection = adapter.get_collection()
    collection.upsert_data([adapter._normalize_record_for_write(_record(1))], ttl=1)
    collection.upsert_data([adapter._normalize_record_for_write(_record(2))], ttl=1)
    collection.upsert_data([adapter._normalize_record_for_write(_record(2))])
    time.sleep(1.5)
    assert [r["id"] for r in adapter.get(["r00001", "r00002"])] == ["r00002"]
    assert adapter.count() == 1


def test_clear_and_delete_by_filter(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    adapter.upsert([_record(i, level=i % 3) for i in range(250)])
    assert adapter.delete(filter=Eq("level", 1)) == 83
    assert adapter.count() == 167 and adapter.count(Eq("level", 1)) == 0
    assert adapter.clear() is True
    assert adapter.count() == 0


def test_read_after_write_on_same_and_other_connection(make_valkey_adapter):
    writer = _create(make_valkey_adapter())
    reader = make_valkey_adapter(name=writer.collection_name)
    misses = []
    for i in range(200):
        writer.upsert(_record(i, uri=f"viking://resources/raw/{i}.md"))
        for adapter in (writer, reader):
            if adapter.count(Eq("uri", f"viking://resources/raw/{i}.md")) != 1:
                misses.append((i, "count"))
            hits = adapter.query(
                query_vector=_record(i)["vector"],
                limit=1,
                filter=Eq("uri", f"viking://resources/raw/{i}.md"),
            )
            if [h["id"] for h in hits] != [f"r{i:05d}"]:
                misses.append((i, "knn"))
        writer.delete(ids=[f"r{i:05d}"])
        if reader.count(Eq("uri", f"viking://resources/raw/{i}.md")) != 0:
            misses.append((i, "delete"))
    assert misses == []


def test_concurrent_writers_and_readers_share_one_adapter(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())

    def work(i):
        adapter.upsert(_record(i))
        assert adapter.get([f"r{i:05d}"])
        return len(adapter.query(query_vector=[1, 0, 0, 0], limit=5))

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(400)))
    assert all(1 <= n <= 5 for n in results)
    assert adapter.count() == 400


async def test_scroll_with_tied_sort_keys_returns_every_record_once(
    make_valkey_adapter, valkey_config
):
    name = _create(make_valkey_adapter()).collection_name
    adapter = create_collection_adapter(valkey_config(name, 4))
    # 1200 records, only 3 distinct updated_at values: offset pages must still be stable.
    adapter.upsert([_record(i, updated_at=f"2026-01-0{1 + i % 3}T00:00:00Z") for i in range(1200)])
    backend = VikingVectorIndexBackend(valkey_config(name, 4))
    settings = SimpleNamespace(vectordb=valkey_config(name, 4), dedicated_vectordb=False)
    backend.set_vector_config_resolver(SimpleNamespace(resolve=AsyncMock(return_value=settings)))
    ctx = RequestContext(user=UserIdentifier("acct", "root"), role=Role.ROOT)
    seen = []
    cursor = None
    try:
        while True:
            page, cursor = await backend.scroll(limit=100, cursor=cursor, ctx=ctx)
            seen += [record["id"] for record in page]
            if cursor is None:
                break
        assert len(seen) == 1200 and len(set(seen)) == 1200
        everything = await backend.filter(filter=Eq("account_id", "acct"), limit=100_000, ctx=ctx)
        assert len(everything) == 1200
    finally:
        await backend.close()
        adapter.close()


def test_sparse_index_creation_is_rejected(make_valkey_adapter):
    adapter = make_valkey_adapter()
    with pytest.raises(NotImplementedError, match="sparse"):
        adapter.create_collection(
            adapter.collection_name,
            CollectionSchemas.context_collection(adapter.collection_name, 4),
            distance="cosine",
            sparse_weight=0.3,
            index_name="default",
        )


def test_sparse_query_vectors_are_ignored_when_dense_present(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    adapter.upsert([_record(i, sparse_vector={"tok": 0.5}) for i in range(20)])
    dense = [1.0, 0.5, 0.2, 0.1]
    plain = adapter.query(query_vector=dense, limit=10)
    hybrid = adapter.query(query_vector=dense, sparse_query_vector={"tok": 1.0}, limit=10)
    assert [r["id"] for r in hybrid] == [r["id"] for r in plain]
    assert [r["_score"] for r in hybrid] == [r["_score"] for r in plain]
    assert adapter.query(sparse_query_vector={"tok": 1.0}, limit=10) == []
    assert adapter.get(["r00001"])[0]["sparse_vector"] == {"tok": 0.5}


def test_unsupported_search_modes_raise(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    with pytest.raises(NotImplementedError):
        adapter.search_by_keywords(query="hello")
    with pytest.raises(NotImplementedError):
        adapter.get_collection().search_by_multimodal("default", text="x", image=None, video=None)


def test_search_by_id_random_and_group_count(make_valkey_adapter):
    adapter = _create(make_valkey_adapter())
    adapter.upsert(
        [_record(i, level=i % 3, context_type=["resource", "memory"][i % 2]) for i in range(30)]
    )
    collection = adapter.get_collection()
    hits = collection.search_by_id("default", "r00004", limit=3)
    assert hits.data[0].id == "r00004" and len(hits.data) == 3
    sample = adapter.search_by_random(filter=Eq("level", 0), limit=50)
    assert len(sample) == 10 and all(r["level"] == 0 for r in sample)
    grouped = collection.aggregate_data("default", field="level")
    assert grouped.agg == {"0": 10, "1": 10, "2": 10}
    grouped = collection.aggregate_data(
        "default", field="context_type", filters={"op": "must", "field": "level", "conds": [0]}
    )
    assert grouped.agg == {"resource": 5, "memory": 5}
    grouped = collection.aggregate_data("default", field="level", cond={"gt": 9, "lt": 11})
    assert grouped.agg == {"0": 10, "1": 10, "2": 10}
    assert collection.aggregate_data("default", field="level", cond={"gt": 10}).agg == {}


def test_scalar_sort_matches_local_and_keeps_records_without_the_field(
    make_valkey_adapter, make_local_adapter
):
    records = []
    for i in range(20):
        record = _record(i, updated_at=f"2026-01-0{1 + i % 3}T00:00:00Z", active_count=i % 4)
        if i % 5 == 0:
            record.pop("updated_at")
        records.append(record)
    local = _create(make_local_adapter(name="sortcmp"))
    valkey = _create(make_valkey_adapter())
    for adapter in (local, valkey):
        adapter.upsert([dict(record) for record in records])
    for desc in (False, True):
        # Without a filter the local sorter leaves out records lacking the field (with a
        # filter it returns them at arbitrary positions). Valkey keeps them, last.
        local_rows = local.query(order_by="updated_at", order_desc=desc, limit=100)
        valkey_rows = valkey.query(order_by="updated_at", order_desc=desc, limit=100)
        assert len(local_rows) == 16 and len(valkey_rows) == 20
        assert [r["updated_at"] for r in valkey_rows[:16]] == [r["updated_at"] for r in local_rows]
        assert sorted(r["id"] for r in valkey_rows[:16]) == sorted(r["id"] for r in local_rows)
        assert {r["updated_at"] for r in valkey_rows[16:]} == {""}
    with pytest.raises(NotImplementedError, match="numeric"):
        valkey.query(order_by="uri", limit=10)
