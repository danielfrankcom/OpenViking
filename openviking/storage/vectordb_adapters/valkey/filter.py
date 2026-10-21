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

from typing import Any, Dict, Mapping

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
    # Open decision (unsupported filter ops): contains/regex need substring matching,
    # which Valkey Search TAG and NUMERIC fields do not offer; geo_range has no Valkey
    # Search field type. They are rejected rather than emulated until maintainers decide.
    raise NotImplementedError(f"Valkey vector backend does not support filter op {op!r}")


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
