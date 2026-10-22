# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Valkey Search result limits, with the server caps lowered: a query that a cap may
have cut must raise a clear error instead of returning partial results."""

import os

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Contains, Eq, Range
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking.storage.vectordb_adapters.valkey.collection import (
    DEFAULT_NONVECTOR_CAP,
    KNN_CAP_SETTING,
    NONVECTOR_CAP_SETTING,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def server_settings(make_valkey_adapter):
    """Change Valkey Search settings on every primary; restore them afterwards."""
    admin = make_valkey_adapter()
    client = admin._client
    target = {"target_nodes": client.PRIMARIES} if hasattr(client, "PRIMARIES") else {}
    original = {}

    def change(name, value):
        if name not in original:
            reply = client.execute_command("CONFIG", "GET", name, **target)
            replies = list(reply.values()) if isinstance(reply, dict) else [reply]
            original[name] = replies[0][1]
        client.execute_command("CONFIG", "SET", name, str(value), **target)

    yield change
    for name, value in original.items():
        client.execute_command("CONFIG", "SET", name, value, **target)


def _filled(adapter, count):
    name = adapter.collection_name
    adapter.create_collection(
        name,
        CollectionSchemas.context_collection(name, 4),
        distance="cosine",
        sparse_weight=0.0,
        index_name="default",
    )
    records = [
        {
            "id": f"r{i:05d}",
            "uri": f"viking://resources/d{i % 10}/f{i}.md",
            "account_id": "acct",
            "active_count": i,
            "updated_at": f"2026-01-{1 + i % 28:02d}T00:00:00Z",
            "vector": [1.0, (i % 13) / 13.0, (i % 7) / 7.0, 0.1],
        }
        for i in range(count)
    ]
    for start in range(0, count, 500):
        adapter.upsert(records[start : start + 500])
    return adapter


def _scroll_ids(adapter, expr=None, page_size=100):
    ids, offset = [], 0
    while True:
        page = adapter.query(
            filter=expr, limit=page_size, offset=offset, order_by="updated_at", order_desc=False
        )
        if not page:
            return ids
        ids += [record["id"] for record in page]
        offset += len(page)


def test_nonvector_cap_raises_instead_of_partial_results(make_valkey_adapter, server_settings):
    # 2000 records put about 667 on each shard of a 3-primary cluster, above the cap.
    server_settings(NONVECTOR_CAP_SETTING, 500)
    adapter = _filled(make_valkey_adapter(), 2000)
    cap_error = NONVECTOR_CAP_SETTING.replace(".", r"\.")

    with pytest.raises(RuntimeError, match=cap_error):
        adapter.count()
    with pytest.raises(RuntimeError, match=cap_error):
        adapter.count(Eq("account_id", "acct"))
    with pytest.raises(RuntimeError, match=cap_error):
        adapter.query(limit=10, order_by="active_count", order_desc=True)
    with pytest.raises(RuntimeError, match=cap_error):
        _scroll_ids(adapter, Eq("account_id", "acct"))
    # Delete-by-filter enumerates with scroll pages first, so it deletes nothing.
    with pytest.raises(RuntimeError, match=cap_error):
        adapter.delete(filter=Eq("account_id", "acct"))
    # contains/regex read their whole superset with non-vector queries.
    with pytest.raises(RuntimeError, match=cap_error):
        adapter.query(filter=Contains("uri", "/d1/"), limit=10, query_vector=[1, 0, 0, 0])

    # Below the cap every path is exact, and the failed delete removed nothing.
    small = Range("active_count", lt=300)
    assert adapter.count(small) == 300
    assert sorted(_scroll_ids(adapter, small)) == [f"r{i:05d}" for i in range(300)]
    assert adapter.count(And([small, Contains("uri", "/d1/")])) == 30
    # KNN queries are not subject to this cap (the KNN cap bounds them).
    assert len(adapter.query(limit=2000)) == 2000


def test_cluster_total_above_cap_is_exact_when_no_shard_reaches_it(
    make_valkey_adapter, server_settings, valkey_params
):
    if valkey_params["mode"] != "cluster":
        pytest.skip("the cap is per shard only in cluster mode")
    # About 667 records per shard, below the cap; the 2000 total is above it.
    server_settings(NONVECTOR_CAP_SETTING, 1000)
    adapter = _filled(make_valkey_adapter(), 2000)
    assert adapter.count() == 2000
    assert len(set(_scroll_ids(adapter, Eq("account_id", "acct"), page_size=500))) == 2000
    assert adapter.count(Contains("uri", "/d1/")) == 200


def test_knn_cap_error_names_the_setting_and_the_k_needed(make_valkey_adapter, server_settings):
    server_settings(KNN_CAP_SETTING, 100)
    adapter = _filled(make_valkey_adapter(), 200)
    knn_error = KNN_CAP_SETTING.replace(".", r"\.")

    with pytest.raises(RuntimeError, match=rf"K=1000: {knn_error} must be at least 1000"):
        adapter.query(limit=1000)
    assert len(adapter.query(limit=100)) == 100
    # A contains filter ranks its whole superset (here all 200 records).
    with pytest.raises(RuntimeError, match=rf"{knn_error} must be at least 200"):
        adapter.query(filter=Contains("uri", "/d1/"), limit=10)
    with pytest.raises(RuntimeError, match="above its maximum of 100000"):
        adapter.query(limit=100001)


def test_startup_reads_a_cap_above_2_31(make_valkey_adapter, server_settings):
    # Valkey Search 1.2.1 reads this value back as -1.
    server_settings(NONVECTOR_CAP_SETTING, 4294967295)
    assert make_valkey_adapter()._nonvector_cap == 4294967295


def test_non_admin_user_assumes_the_default_cap(make_valkey_adapter, valkey_config, valkey_params):
    if valkey_params["mode"] == "cluster":
        pytest.skip("the ACL user is created on one node only")
    admin = _filled(make_valkey_adapter(), 3)
    admin._client.execute_command(
        "ACL", "SETUSER", "ovapp", "on", ">s3cret", "~*", "+@all", "-@admin", "-@dangerous"
    )
    os.environ["OV_VALKEY_TEST_PASSWORD"] = "s3cret"
    acl_params = {"username": "ovapp", "password_env": "OV_VALKEY_TEST_PASSWORD"}
    try:
        adapter = create_collection_adapter(
            valkey_config(admin.collection_name, 4, valkey=acl_params)
        )
        assert adapter._nonvector_cap == DEFAULT_NONVECTOR_CAP
        assert adapter.count() == 3
        assert adapter.count(Contains("uri", "/d1/")) == 1
        adapter.close()
    finally:
        admin._client.execute_command("ACL", "DELUSER", "ovapp")
        os.environ.pop("OV_VALKEY_TEST_PASSWORD", None)
