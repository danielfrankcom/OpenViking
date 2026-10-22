# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Compile OpenViking's filter DSL into a Valkey Search query string.

The input is the dict DSL produced by ``CollectionAdapter._compile_filter`` after
``DataProcessor.convert_filter_for_index`` has turned date_time values into epoch
milliseconds. The output is the query part of ``FT.SEARCH``/``FT.AGGREGATE``.

How each field type is stored (see ``index_values``):

* numeric types become NUMERIC attributes;
* string-like types become TAG attributes, each value wrapped in ``TAG_MARK``;
* a path field ``f`` becomes three attributes: ``f`` (TAG, the record's own path),
  ``f__ancestors`` (TAG, the path and every ancestor) and ``f__depth`` (NUMERIC,
  number of path segments). A directory scope is then "my own path has this
  ancestor", optionally bounded by depth, which never consults any other record.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

TAG_SEPARATOR = "\x1f"
# Valkey Search strips whitespace around TAG values and treats a trailing "*" as a
# prefix query even when it is escaped. Wrapping every stored and queried TAG value
# in this mark keeps leading/trailing spaces and trailing "*" literal, and makes ""
# a real, matchable value.
TAG_MARK = "\x1e"

# Every record stores __all=1. Valkey Search before 1.3 rejects a bare "*" query,
# so this range is the match-all expression for filter-only queries.
ALL_FIELD = "__all"
MATCH_ALL = f"@{ALL_FIELD}:[1 1]"
MATCH_NONE = f"-{MATCH_ALL}"

NUMERIC_TYPES = {"int64", "float32", "date_time", "bool"}
TAG_TYPES = {"string", "text", "list<string>", "list<int64>"}
PATH_TYPE = "path"

# Valkey Search has no substring or regex match on TAG/NUMERIC fields, so these
# clauses are evaluated by the collection itself (see ValkeyCollection._resolve_filter).
CLIENT_SIDE_OPS = ("contains", "regex")
# Filter nodes that compile to "every record" and "no record".
TRUE_NODE: Dict[str, Any] = {"op": "and", "conds": []}
FALSE_NODE: Dict[str, Any] = {"op": "or", "conds": []}


def ancestors_field(field: str) -> str:
    return f"{field}__ancestors"


def depth_field(field: str) -> str:
    return f"{field}__depth"


def path_segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


def canonical_path(path: str) -> str:
    return "/" + "/".join(path_segments(path))


def tag_value(value: Any) -> str:
    return f"{TAG_MARK}{value}{TAG_MARK}"


def number_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return repr(value) if isinstance(value, float) else str(value)


def indexed_hash_fields(field_types: Mapping[str, str]) -> set[str]:
    """Every hash field ``index_values`` can write for these indexed fields."""
    names: set[str] = set()
    for field, field_type in field_types.items():
        names.add(field)
        if field_type == PATH_TYPE:
            names |= {ancestors_field(field), depth_field(field)}
    return names


def index_values(record: Mapping[str, Any], field_types: Mapping[str, str]) -> Dict[str, str]:
    """Return the hash fields Valkey Search indexes for one record.

    ``record`` must already have date_time values converted to epoch milliseconds.
    Fields that are absent (None or an empty list) produce no hash field.
    """
    values: Dict[str, str] = {}
    for field, field_type in field_types.items():
        value = record.get(field)
        if value is None or value == []:
            continue
        if field_type == PATH_TYPE:
            segments = path_segments(str(value))
            ancestors = ["/" + "/".join(segments[:depth]) for depth in range(len(segments) + 1)]
            values[field] = tag_value(canonical_path(str(value)))
            values[ancestors_field(field)] = TAG_SEPARATOR.join(tag_value(a) for a in ancestors)
            values[depth_field(field)] = str(len(segments))
        elif field_type in NUMERIC_TYPES:
            values[field] = number_value(value)
        elif isinstance(value, list):
            values[field] = TAG_SEPARATOR.join(tag_value(item) for item in value)
        else:
            values[field] = tag_value(value)
    return values


def compile_filter(node: Mapping[str, Any] | None, field_types: Mapping[str, str]) -> str:
    """Return a Valkey Search query matching the same records as ``node``."""
    if not node:
        return MATCH_ALL
    op = node.get("op")
    if op in ("and", "or"):
        parts = [compile_filter(child, field_types) for child in node.get("conds", [])]
        if not parts:
            # An empty conjunction is true and an empty disjunction is false.
            return MATCH_ALL if op == "and" else MATCH_NONE
        joiner = " " if op == "and" else " | "
        return "(" + joiner.join(parts) + ")"

    field = node["field"]
    field_type = field_types.get(field)
    if op == "must":
        return _match_any(field, field_type, node.get("conds", []), node.get("para"))
    if op == "must_not":
        return "-" + _match_any(field, field_type, node.get("conds", []), node.get("para"))
    if op == "range":
        return _range(field, node)
    if op == "range_out":
        return "-" + _range(field, node)
    if op == "prefix":
        return f"@{field}:{{{_escape(TAG_MARK + str(node['prefix']))}*}}"
    if op in CLIENT_SIDE_OPS:
        raise ValueError(f"{op!r} clauses must be replaced before compiling the filter")
    # geo_range is rejected: Valkey Search has no geo field type here, no OpenViking
    # schema has a geo_point field, and the local backend rejects geo_range on any
    # other field type ("geo_range field must be geo_point").
    raise NotImplementedError(f"Valkey vector backend does not support filter op {op!r}")


def client_side_clauses(node: Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    """Return the contains/regex clauses of ``node`` in tree order."""
    if not node:
        return []
    if node.get("op") in ("and", "or"):
        return [clause for child in node.get("conds", []) for clause in client_side_clauses(child)]
    return [node] if node.get("op") in CLIENT_SIDE_OPS else []


def replace_client_side_clauses(node: Mapping[str, Any], results: List[bool]) -> Dict[str, Any]:
    """Copy ``node`` with its contains/regex clauses, in tree order, fixed to ``results``."""
    remaining = iter(results)

    def replace(current: Mapping[str, Any]) -> Dict[str, Any]:
        if current.get("op") in ("and", "or"):
            return {**current, "conds": [replace(child) for child in current.get("conds", [])]}
        if current.get("op") in CLIENT_SIDE_OPS:
            return TRUE_NODE if next(remaining) else FALSE_NODE
        return dict(current)

    return replace(node)


def clause_matches(
    clause: Mapping[str, Any], record: Mapping[str, Any], field_type: str | None
) -> bool:
    """Evaluate one contains/regex clause on one record exactly like the local engine.

    ``record`` must already have date_time values converted to epoch milliseconds, and
    ``field_type`` is None when the field is not a scalar index field.

    The local engine (bitmap_field_group.cpp) matches the clause against the keys of
    the field's bitmap index and returns the records under every matching key:

    * only scalar index fields have keys, and float32 fields have none;
    * a key is the value as text, with integers, date_time and bool as decimal
      digits (bool is 1/0), one key per list element, every key split again at ";",
      and path keys given a leading "/";
    * contains is a byte substring test, regex is std::regex_search (ECMAScript) on
      the UTF-8 bytes;
    * an empty substring or pattern, and an invalid pattern, match nothing.
    """
    if field_type is None or field_type == "float32":
        return False
    if clause["op"] == "contains":
        substring = clause.get("substring")
        if not substring:
            return False
        return any(
            substring in key for key in _local_index_keys(record.get(clause["field"]), field_type)
        )
    pattern = clause.get("pattern")
    if not pattern:
        return False
    try:
        # Byte patterns make "." one byte and \w, \d, \s ASCII-only, as in std::regex.
        compiled = re.compile(pattern.encode())
    except re.error:
        return False
    keys = _local_index_keys(record.get(clause["field"]), field_type)
    return any(compiled.search(key.encode()) for key in keys)


def _local_index_keys(value: Any, field_type: str) -> List[str]:
    if value is None:
        return []
    keys = []
    for item in value if isinstance(value, list) else [value]:
        text = str(int(item)) if isinstance(item, (bool, int)) else str(item)
        for key in text.split(";"):
            if field_type == PATH_TYPE and key and not key.startswith("/"):
                key = "/" + key
            keys.append(key)
    return keys


def _match_any(field: str, field_type: str | None, conds: list[Any], para: Any) -> str:
    if not conds:
        return MATCH_NONE
    if field_type == PATH_TYPE:
        depth = _parse_depth(para)
        parts = [_path_scope(field, str(path), depth) for path in conds]
    elif field_type in NUMERIC_TYPES:
        parts = [f"@{field}:[{number_value(v)} {number_value(v)}]" for v in conds]
    else:
        tags = "|".join(_escape(tag_value(v)) for v in conds)
        return f"@{field}:{{{tags}}}"
    return parts[0] if len(parts) == 1 else "(" + " | ".join(parts) + ")"


def _path_scope(field: str, path: str, depth: int) -> str:
    """Match records whose own path is ``path`` or lies at most ``depth`` levels below it."""
    path = canonical_path(path)
    if depth == 0:
        return f"@{field}:{{{_escape(tag_value(path))}}}"
    if depth < 0 and path == "/":
        # The local engine treats an unbounded scope at the root as "everything".
        return MATCH_ALL
    within_subtree = f"@{ancestors_field(field)}:{{{_escape(tag_value(path))}}}"
    if depth < 0:
        return within_subtree
    base_depth = len(path_segments(path))
    return f"({within_subtree} @{depth_field(field)}:[{base_depth} {base_depth + depth}])"


def _parse_depth(para: Any) -> int:
    # Same convention as the local engine: "-d=N" clamped to [-1, 50], absent means
    # unbounded (-1).
    if not para:
        return -1
    text = "".join(str(para).split())
    if not text.startswith("-d="):
        return -1
    return max(-1, min(50, int(text[3:])))


def _range(field: str, node: Mapping[str, Any]) -> str:
    low, high = "-inf", "+inf"
    if node.get("gte") is not None:
        low = number_value(node["gte"])
    elif node.get("gt") is not None:
        low = "(" + number_value(node["gt"])
    if node.get("lte") is not None:
        high = number_value(node["lte"])
    elif node.get("lt") is not None:
        high = "(" + number_value(node["lt"])
    return f"@{field}:[{low} {high}]"


def _escape(text: str) -> str:
    # Backslash-escape ASCII punctuation and spaces so "|", "}" and "\" inside a value
    # cannot end the TAG set. Valkey Search unescapes any "\x" back to "x".
    return "".join("\\" + ch if ch.isascii() and not ch.isalnum() else ch for ch in text)
