# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Differential filter tests: every filter must select the same records on Valkey as
on OpenViking's local engine (the reference implementation of the DSL)."""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.acl import ACL_GRANT_FIELDS
from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import (
    And,
    Contains,
    Eq,
    In,
    Or,
    PathScope,
    Range,
    RawDSL,
    TimeRange,
)
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
from openviking.utils.tags import build_search_tags_filter
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

pytestmark = pytest.mark.integration

URIS = [
    "viking://resources",
    "viking://resources/a",
    "viking://resources/a/.abstract.md",
    "viking://resources/a/b.md",
    "viking://resources/a/b.md#chunk_0001",
    "viking://resources/a/c",
    "viking://resources/a/c/d.md",
    "viking://resources/a/c/e/f/g.md",
    "viking://resources/ab/sibling.md",
    "viking://resources/sp ace/x,y{z}|w.md",
    "viking://resources/star/name*",
    "viking://resources/star/name*more",
    "viking://resources/中文/文件.md",
    "viking://user/alice/memories/m1.md",
    "viking://user/alice/memories/events/e1.md",
    "viking://user/bob/memories/m2.md",
    "viking://agent/skills/s1",
    "viking://",
    # After URI encoding this stays "/resources/x/viking://resources/a/b.md#weird", so a
    # Contains with an unencoded "viking://..." substring matches it on the local engine.
    "viking://resources/x/viking://resources/a/b.md#weird",
]


def _record(index: int, uri: str):
    level = 0 if uri.endswith(".abstract.md") else (1 if "." not in uri.rsplit("/", 1)[-1] else 2)
    context_type = (
        "memory" if "/memories/" in uri else ("skill" if "/skills/" in uri else "resource")
    )
    tags = [f"k{index % 3}=v", f"group={'odd' if index % 2 else 'even'}"]
    if "/events/" in uri:
        tags.append("memory_type=events")
    acl_modes = ["none", "inherit", "restricted"]
    record = {
        "id": f"r{index:02d}",
        "uri": uri,
        "level": level,
        "context_type": context_type,
        "account_id": "acct" if index % 5 else "other",
        "owner_user_id": "alice" if "alice" in uri else "bob",
        "name": "" if index % 4 == 0 else f"name{index}",
        "active_count": index % 7,
        "updated_at": f"2026-01-{1 + index // 3:02d}T00:00:00Z",
        "created_at": f"2025-12-{1 + index:02d}T08:30:00+08:00",
        "search_tags": tags,
        "acl_mode": acl_modes[index % 3],
        "acl_direct_grants": [f"read:user:u{index % 4}"],
        "acl_inherited_grants": [f"read:user:u{(index + 1) % 4}"] if index % 2 else [],
        "vector": [1.0, float(index % 5), float(index % 3), 0.5],
    }
    if index % 6 == 0:
        # Leave some optional ACL fields to their schema defaults.
        for field in ("acl_mode", *ACL_GRANT_FIELDS):
            record.pop(field)
    return record


CORPUS = [_record(i, uri) for i, uri in enumerate(URIS)]
CORPUS[3]["name"] = " padded "  # leading/trailing spaces must stay significant
CORPUS[4]["name"] = "padded"
CORPUS[5]["name"] = "semi;colon"  # the local engine indexes "semi" and "colon" separately


def _ctx(role: Role, user: str = "alice") -> RequestContext:
    return RequestContext(user=UserIdentifier("acct", user), role=role)


def _scope_filter(ctx, *, acl_enabled, targets=None, level=None, context_type=None):
    backend = object.__new__(VikingVectorIndexBackend)
    backend.acl_manager = None
    return backend._build_scope_filter(
        ctx=ctx,
        context_type=context_type,
        target_directories=targets,
        extra_filter=None,
        level=level,
        acl_enabled=acl_enabled,
    )


FILTERS = {
    "none": None,
    "level_eq": Eq("level", 2),
    "level_in": In("level", [0, 1]),
    "level_in_empty": In("level", []),
    "context_type_in": In("context_type", ["memory", "skill"]),
    "account": Eq("account_id", "acct"),
    "name_empty_string": Eq("name", ""),
    "name_with_spaces": Eq("name", " padded "),
    "name_without_spaces": Eq("name", "padded"),
    "uri_eq_dir": Eq("uri", "viking://resources/a"),
    "uri_eq_file": Eq("uri", "viking://resources/a/b.md"),
    "uri_eq_special": Eq("uri", "viking://resources/sp ace/x,y{z}|w.md"),
    "uri_eq_star": Eq("uri", "viking://resources/star/name*"),
    "uri_eq_unicode": Eq("uri", "viking://resources/中文/文件.md"),
    "scope_root_all": PathScope("uri", "viking://", depth=-1),
    "scope_root_1": PathScope("uri", "viking://", depth=1),
    "scope_resources_all": PathScope("uri", "viking://resources", depth=-1),
    "scope_a_all": PathScope("uri", "viking://resources/a", depth=-1),
    "scope_a_0": PathScope("uri", "viking://resources/a", depth=0),
    "scope_a_1": PathScope("uri", "viking://resources/a", depth=1),
    "scope_a_2": PathScope("uri", "viking://resources/a", depth=2),
    "scope_a_trailing_slash": PathScope("uri", "viking://resources/a/", depth=-1),
    "scope_file_all": PathScope("uri", "viking://resources/a/b.md", depth=-1),
    "scope_missing": PathScope("uri", "viking://resources/nope", depth=-1),
    "scope_special_dir": PathScope("uri", "viking://resources/sp ace", depth=1),
    "raw_must_uri_no_para": RawDSL(
        {"op": "must", "field": "uri", "conds": ["viking://resources/a/c"]}
    ),
    "raw_must_not_subtree": RawDSL(
        {"op": "must_not", "field": "uri", "conds": ["viking://resources/a"], "para": "-d=-1"}
    ),
    "raw_must_multi_paths": RawDSL(
        {
            "op": "must",
            "field": "uri",
            "conds": ["viking://user/alice", "viking://agent"],
            "para": "-d=-1",
        }
    ),
    "tag_one": In("search_tags", ["k1=v"]),
    "tags_all_of": RawDSL(build_search_tags_filter(["k1=v", "group=odd"])),
    "tags_any_of": In("search_tags", ["k0=v", "memory_type=events"]),
    "tags_must_not": RawDSL(
        {"op": "must_not", "field": "search_tags", "conds": ["memory_type=events"]}
    ),
    "grants_in": In("acl_direct_grants", ["read:user:u1", "read:user:u2"]),
    "acl_mode_must_not": RawDSL(
        {"op": "must_not", "field": "acl_mode", "conds": ["inherit", "restricted"]}
    ),
    "active_range": Range("active_count", gte=1, lt=5),
    "active_range_gt_lte": Range("active_count", gt=2, lte=6),
    "active_range_out": RawDSL({"op": "range_out", "field": "active_count", "gte": 2, "lte": 4}),
    "updated_time_range": TimeRange(
        "updated_at", start="2026-01-02T00:00:00Z", end="2026-01-04T00:00:00Z"
    ),
    "raw_time_range": RawDSL(
        {
            "op": "time_range",
            "field": "updated_at",
            "gte": "2026-01-03T00:00:00+00:00",
            "lte": "2026-01-05T00:00:00Z",
        }
    ),
    "created_time_range_tz": TimeRange("created_at", start="2025-12-05T00:00:00Z"),
    "uri_prefix": RawDSL({"op": "prefix", "field": "uri", "prefix": "viking://resources/a"}),
    "and_nested": And(
        [
            PathScope("uri", "viking://resources", depth=-1),
            Or([Eq("level", 0), In("search_tags", ["k2=v"])]),
        ]
    ),
    "or_paths": Or(
        [
            PathScope("uri", "viking://resources/a/c", depth=0),
            PathScope("uri", "viking://user", depth=-1),
        ]
    ),
    "scope_user_no_acl": _scope_filter(_ctx(Role.USER), acl_enabled=False),
    "scope_user_acl": _scope_filter(_ctx(Role.USER), acl_enabled=True),
    "scope_admin_acl": _scope_filter(_ctx(Role.ADMIN), acl_enabled=True),
    "scope_root": _scope_filter(_ctx(Role.ROOT), acl_enabled=True),
    "scope_user_targets_level": _scope_filter(
        _ctx(Role.USER), acl_enabled=False, targets=["viking://resources/a"], level=[0, 1]
    ),
    "scope_user_memory_type": _scope_filter(
        _ctx(Role.USER), acl_enabled=False, context_type="memory"
    ),
}


@pytest.fixture(scope="module")
def adapters(tmp_path_factory, valkey_config):
    schema = CollectionSchemas.context_collection("difftest", 4)
    local = create_collection_adapter(
        VectorDBBackendConfig(
            backend="local",
            name="difftest",
            dimension=4,
            path=str(tmp_path_factory.mktemp("local")),
        )
    )
    valkey = create_collection_adapter(valkey_config("difftest", 4))
    valkey.drop_collection()
    for adapter in (local, valkey):
        adapter.create_collection(
            "difftest", schema, distance="cosine", sparse_weight=0.0, index_name="default"
        )
        adapter.upsert([dict(record) for record in CORPUS])
    yield local, valkey
    valkey.drop_collection()
    valkey.close()
    local.close()


def _ids(records):
    return sorted(record["id"] for record in records)


@pytest.mark.parametrize("name", sorted(FILTERS))
def test_filter_selects_same_records_as_local(adapters, name):
    local, valkey = adapters
    expr = FILTERS[name]
    local_ids = _ids(local.query(filter=expr, limit=1000, query_vector=[1.0, 0.0, 0.0, 0.0]))
    valkey_ids = _ids(valkey.query(filter=expr, limit=1000, query_vector=[1.0, 0.0, 0.0, 0.0]))
    assert valkey_ids == local_ids
    assert valkey.count(expr) == local.count(expr) == len(local_ids)


@pytest.mark.parametrize(
    "name", ["none", "scope_a_all", "tags_all_of", "scope_user_acl", "active_range"]
)
def test_filter_only_and_sorted_queries_match_local(adapters, name):
    local, valkey = adapters
    expr = FILTERS[name]
    # Filter-only queries use a random vector; sorted queries use search_by_scalar.
    assert _ids(valkey.query(filter=expr, limit=1000)) == _ids(local.query(filter=expr, limit=1000))
    local_sorted = local.query(filter=expr, limit=1000, order_by="active_count", order_desc=True)
    valkey_sorted = valkey.query(filter=expr, limit=1000, order_by="active_count", order_desc=True)
    assert [r["active_count"] for r in valkey_sorted] == [r["active_count"] for r in local_sorted]
    assert _ids(valkey_sorted) == _ids(local_sorted)


def _contains(field, substring):
    return RawDSL({"op": "contains", "field": field, "substring": substring})


def _regex(field, pattern):
    return RawDSL({"op": "regex", "field": field, "pattern": pattern})


def _legacy_cleanup_filter(uri):
    """The filter vector_migration._records_in_scope sends (recursive, not volcengine)."""
    return And(
        [
            Eq("account_id", "acct"),
            Or([Eq("uri", uri), PathScope("uri", uri, depth=-1), Contains("uri", uri + "#")]),
        ]
    )


# contains/regex are evaluated by the Valkey backend itself; every case must select
# exactly what the local engine selects, including its quirks (unencoded viking://
# substrings, ";"-split keys, integer keys, byte-wise regex, empty patterns).
CLIENT_SIDE_FILTERS = {
    "contains_uri_encoded": Contains("uri", "/resources/a/"),
    "contains_uri_chunk": Contains("uri", "b.md#"),
    "contains_uri_unencoded": Contains("uri", "viking://resources/a/b.md#"),
    "contains_uri_unicode": Contains("uri", "中文"),
    "contains_name": _contains("name", "name1"),
    "contains_name_spaces": _contains("name", " padded"),
    "contains_name_semicolon_split": _contains("name", "colon"),
    "contains_name_across_semicolon": _contains("name", "semi;colon"),
    "contains_tags": _contains("search_tags", "=odd"),
    "contains_int": _contains("active_count", "3"),
    "contains_date_time_digits": _contains("updated_at", "1767"),
    "contains_date_time_text": _contains("updated_at", "2026"),
    "contains_grants": _contains("acl_inherited_grants", "u2"),
    # md5 is stored (default "") but not a scalar index field, so it has no keys.
    "regex_unindexed_field": _regex("md5", "^$"),
    "contains_missing_field": _contains("no_such_field", "x"),
    "contains_empty": _contains("name", ""),
    "regex_uri_anchor": _regex("uri", "^/resources/a/"),
    "regex_uri_suffix": _regex("uri", "\\.md$"),
    "regex_uri_unencoded": _regex("uri", "^viking://"),
    "regex_name_alternation": _regex("name", "^(name1|padded)"),
    # std::regex works on bytes: "." is one byte (文件 is 6) and \w is ASCII-only.
    "regex_dot_is_one_byte": _regex("uri", "/.{6}\\.md$"),
    "regex_word_class_is_ascii": _regex("uri", "/\\w+\\.md$"),
    "regex_word_class": _regex("name", "^\\w+$"),
    "regex_tags": _regex("search_tags", "^k[12]="),
    "regex_int": _regex("level", "^[01]$"),
    "regex_empty_string_value": _regex("name", "^$"),
    "regex_invalid": _regex("name", "("),
    "regex_empty_pattern": _regex("name", ""),
    "and_scope_contains": And(
        [PathScope("uri", "viking://resources", depth=-1), Contains("uri", ".md")]
    ),
    "or_eq_contains": Or([Eq("level", 0), Contains("uri", "/memories/")]),
    "or_must_not_contains": Or(
        [
            RawDSL({"op": "must_not", "field": "acl_mode", "conds": ["inherit"]}),
            _contains("name", "1"),
        ]
    ),
    "or_contains_unmatched": Or([Eq("level", 1), Contains("uri", "nothing-has-this")]),
    "nested_two_clauses": And(
        [
            Or([Contains("uri", "alice"), _contains("search_tags", "odd")]),
            Or([Range("active_count", gte=2), _regex("name", "^name1")]),
        ]
    ),
    "acl_scope_and_regex": And(
        [_scope_filter(_ctx(Role.USER), acl_enabled=True), _regex("uri", "m[12]\\.md$")]
    ),
    "legacy_cleanup_file": _legacy_cleanup_filter("viking://resources/a/b.md"),
    "legacy_cleanup_dir": _legacy_cleanup_filter("viking://resources/a"),
    "legacy_cleanup_session": _legacy_cleanup_filter("viking://session"),
}


def _scroll_ids(adapter, expr, page_size=4):
    """Page through every match the way delete-by-filter and ACL scans do."""
    ids, offset = [], 0
    while True:
        page = adapter.query(
            filter=expr, limit=page_size, offset=offset, order_by="updated_at", order_desc=False
        )
        if not page:
            return ids
        ids += [record["id"] for record in page]
        offset += len(page)


@pytest.mark.parametrize("name", sorted(CLIENT_SIDE_FILTERS))
def test_contains_and_regex_select_same_records_as_local(adapters, name):
    local, valkey = adapters
    expr = CLIENT_SIDE_FILTERS[name]
    query = [1.0, 0.0, 0.0, 0.0]
    local_ids = _ids(local.query(filter=expr, limit=1000, query_vector=query))
    # Vector search, filter-only search (random vector), counts and scroll pages.
    assert _ids(valkey.query(filter=expr, limit=1000, query_vector=query)) == local_ids
    assert _ids(valkey.query(filter=expr, limit=1000)) == _ids(local.query(filter=expr, limit=1000))
    assert valkey.count(expr) == local.count(expr) == len(local_ids)
    scrolled = _scroll_ids(valkey, expr)
    assert sorted(scrolled) == local_ids == sorted(_scroll_ids(local, expr))
    # A top-k page keeps vector order: compare scores, as tied records may swap.
    local_page = local.query(filter=expr, limit=3, offset=1, query_vector=query)
    valkey_page = valkey.query(filter=expr, limit=3, offset=1, query_vector=query)
    assert [hit["_score"] for hit in valkey_page] == pytest.approx(
        [hit["_score"] for hit in local_page], abs=0.02
    )
    # A sorted page keeps the sort order of the whole match set.
    local_sorted = local.query(filter=expr, limit=3, offset=1, order_by="active_count")
    valkey_sorted = valkey.query(filter=expr, limit=3, offset=1, order_by="active_count")
    assert [r["active_count"] for r in valkey_sorted] == [r["active_count"] for r in local_sorted]


def test_legacy_cleanup_filter_finds_what_local_finds(adapters):
    """The only internal caller's Contains term is sent unencoded, so on the local engine
    it misses the encoded chunk record (r04) and matches only a URI that literally
    contains "viking://resources/a/b.md#" (r18). Valkey must do the same."""
    local, valkey = adapters
    expr = _legacy_cleanup_filter("viking://resources/a/b.md")
    assert _ids(local.query(filter=expr, limit=100000)) == ["r03", "r18"]
    assert _ids(valkey.query(filter=expr, limit=100000)) == ["r03", "r18"]


def test_empty_substring_in_or_drops_earlier_matches_on_local_only(adapters):
    """Documented divergence: on the local engine an empty substring or pattern returns
    'no bitmap', and inside an or-node that discards the matches of the clauses before
    it (filter_ops.cpp ContainsOp::calc_bitmap). Valkey treats the clause as false. No
    OpenViking caller sends an empty substring."""
    local, valkey = adapters
    expr = Or([Eq("level", 0), _contains("name", "")])
    assert local.count(expr) == 0
    assert valkey.count(expr) == local.count(Eq("level", 0)) > 0


def test_contains_in_grouped_count_is_rejected(adapters):
    _, valkey = adapters
    collection = valkey.get_collection()
    with pytest.raises(NotImplementedError, match="grouped count"):
        collection.aggregate_data(
            "default",
            op="count",
            field="level",
            filters={"op": "contains", "field": "uri", "substring": "a"},
        )


def test_records_round_trip_identically(adapters):
    local, valkey = adapters
    ids = [record["id"] for record in CORPUS]
    local_records = {r["id"]: r for r in local.get(ids)}
    valkey_records = {r["id"]: r for r in valkey.get(ids)}
    assert valkey_records.keys() == local_records.keys()
    for record_id, local_record in local_records.items():
        valkey_record = dict(valkey_records[record_id])
        local_record = dict(local_record)
        assert valkey_record.pop("vector") == pytest.approx(local_record.pop("vector"))
        assert valkey_record == local_record, record_id


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "or", "conds": []},
        {
            "op": "and",
            "conds": [{"op": "must", "field": "level", "conds": [2]}, {"op": "and", "conds": []}],
        },
    ],
)
def test_empty_logic_nodes_are_strict_where_local_fails_open(adapters, payload):
    """Documented divergence: the local engine rejects an empty and/or node and then
    ignores the whole filter (every record matches). Valkey evaluates the node
    (and([]) is true, or([]) is false). No OpenViking caller emits these nodes:
    base._compile_filter collapses empty And/Or and _merge_filters drops empty and."""
    local, valkey = adapters
    assert local.count(RawDSL(payload)) == len(CORPUS)
    expected = 0 if payload["op"] == "or" else local.count(Eq("level", 2))
    assert valkey.count(RawDSL(payload)) == expected
