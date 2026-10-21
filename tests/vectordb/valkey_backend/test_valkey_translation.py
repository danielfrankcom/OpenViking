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
    compile_filter,
    index_values,
)

FIELD_TYPES = {"uri": "path", "level": "int64", "name": "string", "search_tags": "list<string>"}


@pytest.mark.parametrize(
    "node",
    [
        {"op": "contains", "field": "uri", "substring": "b.md#"},
        {"op": "regex", "field": "name", "pattern": "a.*"},
        {"op": "geo_range", "field": "location", "center": [0, 0], "radius": 1},
    ],
)
def test_unsupported_filter_ops_are_rejected(node):
    with pytest.raises(NotImplementedError, match=node["op"]):
        compile_filter({"op": "and", "conds": [node]}, FIELD_TYPES)


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
