# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Filter DSL -> Valkey Search query and distance -> score translation (no server needed).

The live tests in this directory check that the compiled queries select the same
records as the local engine; these pin the rules that hold without a server.
"""

import pytest

from openviking.storage.vectordb_adapters.valkey.collection import _score_from_distance
from openviking.storage.vectordb_adapters.valkey.filter import (
    MATCH_ALL,
    MATCH_NONE,
    TAG_MARK,
    TAG_SEPARATOR,
    clause_matches,
    client_side_clauses,
    compile_filter,
    index_values,
    replace_client_side_clauses,
)

FIELD_TYPES = {"uri": "path", "level": "int64", "name": "string", "search_tags": "list<string>"}


def test_geo_range_is_rejected():
    node = {"op": "geo_range", "field": "location", "center": [0, 0], "radius": 1}
    with pytest.raises(NotImplementedError, match="geo_range"):
        compile_filter({"op": "and", "conds": [node]}, FIELD_TYPES)


def test_contains_and_regex_compile_only_after_replacement():
    contains = {"op": "contains", "field": "uri", "substring": "b.md#"}
    regex = {"op": "regex", "field": "name", "pattern": "a.*"}
    node = {"op": "or", "conds": [{"op": "must", "field": "level", "conds": [1]}, contains, regex]}
    assert client_side_clauses(node) == [contains, regex]
    with pytest.raises(ValueError, match="must be replaced"):
        compile_filter(node, FIELD_TYPES)
    assert compile_filter(replace_client_side_clauses(node, [True, False]), FIELD_TYPES) == (
        f"(@level:[1 1] | {MATCH_ALL} | {MATCH_NONE})"
    )


@pytest.mark.parametrize(
    ("clause", "record", "field_type", "expected"),
    [
        ({"op": "contains", "substring": "b.md#"}, {"f": "/a/b.md#c"}, "path", True),
        # Stored URIs are encoded; an unencoded substring matches only a literal one.
        ({"op": "contains", "substring": "viking://a"}, {"f": "/a"}, "path", False),
        ({"op": "contains", "substring": "/a"}, {"f": "a"}, "path", True),
        ({"op": "contains", "substring": "x;y"}, {"f": "x;y"}, "string", False),
        ({"op": "contains", "substring": "y"}, {"f": ["k", "x;y"]}, "list<string>", True),
        ({"op": "contains", "substring": "17"}, {"f": 1767225600000}, "date_time", True),
        ({"op": "contains", "substring": "1"}, {"f": True}, "bool", True),
        ({"op": "contains", "substring": "1"}, {"f": 1.5}, "float32", False),
        ({"op": "contains", "substring": "a"}, {"f": "a"}, None, False),
        ({"op": "contains", "substring": ""}, {"f": "a"}, "string", False),
        ({"op": "regex", "pattern": "^$"}, {"f": ""}, "string", True),
        ({"op": "regex", "pattern": "^$"}, {}, "string", False),
        ({"op": "regex", "pattern": "^.$"}, {"f": "é"}, "string", False),
        ({"op": "regex", "pattern": "^..$"}, {"f": "é"}, "string", True),
        ({"op": "regex", "pattern": "^\\w$"}, {"f": "é"}, "string", False),
        ({"op": "regex", "pattern": "("}, {"f": "("}, "string", False),
        ({"op": "regex", "pattern": ""}, {"f": "a"}, "string", False),
    ],
)
def test_clauses_match_like_the_local_engine_index_keys(clause, record, field_type, expected):
    assert clause_matches({**clause, "field": "f"}, record, field_type) is expected


def test_empty_logic_nodes_are_evaluated():
    # The local engine ignores a filter containing an empty and/or node; Valkey
    # evaluates it. No OpenViking caller emits one.
    assert compile_filter(None, FIELD_TYPES) == MATCH_ALL
    assert compile_filter({"op": "and", "conds": []}, FIELD_TYPES) == MATCH_ALL
    assert compile_filter({"op": "or", "conds": []}, FIELD_TYPES) == MATCH_NONE


@pytest.mark.parametrize(
    ("para", "expected"),
    [
        ("-d=0", "@uri:{\\\x1e\\/a\\/b\\\x1e}"),
        (None, "@uri__ancestors:{\\\x1e\\/a\\/b\\\x1e}"),
        ("-d=-1", "@uri__ancestors:{\\\x1e\\/a\\/b\\\x1e}"),
        ("-d=1", "(@uri__ancestors:{\\\x1e\\/a\\/b\\\x1e} @uri__depth:[2 3])"),
        ("-d=99", "(@uri__ancestors:{\\\x1e\\/a\\/b\\\x1e} @uri__depth:[2 52])"),
    ],
)
def test_path_scope_uses_only_the_records_own_path(para, expected):
    node = {"op": "must", "field": "uri", "conds": ["/a/b/"], "para": para}
    assert compile_filter(node, FIELD_TYPES) == expected


def test_path_values_index_every_ancestor_and_the_depth():
    values = index_values({"uri": "/a/b c/d.md"}, FIELD_TYPES)
    ancestors = ["/", "/a", "/a/b c", "/a/b c/d.md"]
    assert values["uri__ancestors"] == TAG_SEPARATOR.join(
        f"{TAG_MARK}{a}{TAG_MARK}" for a in ancestors
    )
    assert values["uri__depth"] == "3"


def test_tag_values_with_query_syntax_stay_literal():
    query = compile_filter({"op": "must", "field": "name", "conds": [" a|b}*", ""]}, FIELD_TYPES)
    assert query == "@name:{\\\x1e\\ a\\|b\\}\\*\\\x1e|\\\x1e\\\x1e}"


@pytest.mark.parametrize(
    ("metric", "distance", "score"),
    [
        ("cosine", 0.0, 1.0),
        ("cosine", 1.0, 0.5),
        ("cosine", 2.0, 0.0),
        ("cosine", 2.0000002, 0.0),
        ("ip", 0.0, 1.0),
        ("ip", 2.0, -1.0),
        ("l2", 0.0, 1.0),
        ("l2", 0.25, 0.75),
    ],
)
def test_distance_converts_to_the_local_engine_score_scale(metric, distance, score):
    assert _score_from_distance(metric, distance) == pytest.approx(score)
