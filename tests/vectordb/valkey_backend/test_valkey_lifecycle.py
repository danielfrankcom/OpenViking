# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Factory routing, collection lifecycle, indexes and metadata persistence on Valkey."""

import json
import os

import pytest
from redis.exceptions import AuthenticationError, ResponseError

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import Eq
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking.storage.vectordb_adapters.valkey_adapter import ValkeyCollectionAdapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config

pytestmark = pytest.mark.integration


def _create(adapter, dimension=4):
    name = adapter.collection_name
    return adapter.create_collection(
        name,
        CollectionSchemas.context_collection(name, dimension),
        distance="cosine",
        sparse_weight=0.0,
        index_name="default",
    )


def _record_keys(adapter):
    collection = adapter.get_collection()._Collection__collection
    return [key for batch in collection._scan_record_keys() for key in batch]


def test_factory_routes_valkey_backend_from_ov_conf(tmp_path, valkey_params):
    vectordb = {
        "backend": "valkey",
        "name": "factory_test",
        "index_name": "factory_index",
        "valkey": {**valkey_params, "index_type": "hnsw", "vector_index_params": {"M": 8}},
    }
    config_path = tmp_path / "ov.conf"
    config_path.write_text(
        json.dumps(
            {
                "storage": {"vectordb": vectordb},
                "embedding": {
                    "dense": {"provider": "openai", "model": "m", "api_key": "k", "dimension": 8}
                },
            }
        )
    )
    try:
        OpenVikingConfigSingleton.initialize(config_path=str(config_path))
        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        assert isinstance(adapter, ValkeyCollectionAdapter)
        assert adapter.mode == "valkey"
        assert (adapter.collection_name, adapter.index_name) == ("factory_test", "factory_index")
        assert adapter._config.index_type == "hnsw"
        assert adapter.collection_exists() is False
        adapter.close()
    finally:
        OpenVikingConfigSingleton.reset_instance()


def test_acl_username_and_password_env(make_valkey_adapter, valkey_config, valkey_params):
    if valkey_params["mode"] == "cluster":
        pytest.skip("the ACL user is created on one node only")
    admin = make_valkey_adapter()
    admin._client.execute_command(
        "ACL", "SETUSER", "ovtest", "on", ">s3cret", "resetkeys", "~ovvec*", "+@all"
    )
    os.environ["OV_VALKEY_TEST_PASSWORD"] = "s3cret"
    acl_params = {"username": "ovtest", "password_env": "OV_VALKEY_TEST_PASSWORD"}
    try:
        adapter = create_collection_adapter(
            valkey_config(admin.collection_name, 4, valkey=acl_params)
        )
        assert _create(adapter)
        adapter.upsert({"id": "a", "uri": "viking://resources/a", "vector": [1, 0, 0, 0]})
        assert adapter.count() == 1
        assert admin.count() == 1
        adapter.close()
        os.environ["OV_VALKEY_TEST_PASSWORD"] = "wrong"
        with pytest.raises(AuthenticationError):
            create_collection_adapter(valkey_config(admin.collection_name, 4, valkey=acl_params))
    finally:
        admin._client.execute_command("ACL", "DELUSER", "ovtest")
        os.environ.pop("OV_VALKEY_TEST_PASSWORD", None)


def test_lifecycle_exists_create_drop(make_valkey_adapter):
    adapter = make_valkey_adapter()
    assert adapter.collection_exists() is False
    assert _create(adapter) is True
    assert _create(adapter) is False
    assert adapter.collection_exists() is True
    adapter.upsert(
        [
            {"id": f"r{i}", "uri": f"viking://resources/{i}", "vector": [1, 0, 0, i]}
            for i in range(5)
        ]
    )
    assert len(_record_keys(adapter)) == 5
    collection = adapter.get_collection()._Collection__collection
    assert adapter.drop_collection() is True
    assert adapter.collection_exists() is False
    assert [key for batch in collection._scan_record_keys() for key in batch] == []
    assert adapter._client.get(f"ovvec-meta:{adapter.collection_name}") is None
    assert adapter.drop_collection() is False


def test_metadata_and_data_survive_reconnect(make_valkey_adapter):
    first = make_valkey_adapter(dimension=8)
    assert _create(first, dimension=8)
    first.upsert({"id": "r1", "uri": "viking://resources/x.md", "level": 2, "vector": [0.1] * 8})
    first.get_collection().update(description="described once")
    original_meta = first.get_collection_info()
    original_index = first.get_collection().get_index_meta_data("default")

    second = make_valkey_adapter(dimension=8, name=first.collection_name)
    assert second.collection_exists()
    assert second.get_collection_info() == original_meta
    assert second.get_collection_info()["Description"] == "described once"
    assert second.get_collection().list_indexes() == ["default"]
    assert second.get_collection().get_index_meta_data("default") == original_index
    assert [r["id"] for r in second.query(filter=Eq("level", 2), limit=5)] == ["r1"]
    assert second.get(["r1"])[0]["vector"] == pytest.approx([0.1] * 8)


def test_schema_update_appends_fields(make_valkey_adapter):
    adapter = make_valkey_adapter()
    _create(adapter)
    adapter.get_collection().update(fields=[{"FieldName": "extra", "FieldType": "string"}])
    names = [f["FieldName"] for f in adapter.get_collection_info()["Fields"]]
    assert names[-1] == "extra" and "uri" in names
    adapter.upsert(
        {"id": "x", "uri": "viking://resources/x", "extra": "hello", "vector": [1, 0, 0, 0]}
    )
    assert adapter.get(["x"])[0]["extra"] == "hello"


def test_update_index_makes_existing_records_filterable(make_valkey_adapter):
    adapter = make_valkey_adapter()
    _create(adapter)
    adapter.upsert(
        [
            {"id": "a", "uri": "viking://resources/a", "md5": "aaa", "vector": [1, 0, 0, 0]},
            {"id": "b", "uri": "viking://resources/b", "md5": "bbb", "vector": [0, 1, 0, 0]},
        ]
    )
    collection = adapter.get_collection()
    with pytest.raises(ResponseError, match="md5"):
        adapter.count(Eq("md5", "aaa"))  # md5 is not a scalar index field yet
    scalar_index = collection.get_index_meta_data("default")["ScalarIndex"]
    collection.update_index("default", scalar_index=[*scalar_index, "md5"])
    assert adapter.count(Eq("md5", "aaa")) == 1
    assert "md5" in collection.get_index_meta_data("default")["ScalarIndex"]


def test_index_lifecycle(make_valkey_adapter):
    adapter = make_valkey_adapter()
    _create(adapter)
    collection = adapter.get_collection()
    assert collection.has_index("default") and collection.list_indexes() == ["default"]
    assert collection.get_index("default")["VectorIndex"]["IndexType"] == "flat"
    collection.create_index(
        "second",
        {
            "VectorIndex": {"IndexType": "hnsw", "Distance": "ip", "Params": {"M": 8}},
            "ScalarIndex": ["level"],
        },
    )
    adapter.upsert({"id": "a", "uri": "viking://resources/a", "level": 1, "vector": [1, 0, 0, 0]})
    hits = collection.search_by_vector(
        "second", dense_vector=[1, 0, 0, 0], filters={"op": "must", "field": "level", "conds": [1]}
    )
    assert [hit.id for hit in hits.data] == ["a"]
    collection.drop_index("second")
    assert collection.list_indexes() == ["default"] and collection.get_index("second") is None


def test_non_default_database_is_isolated(make_valkey_adapter, valkey_params):
    if valkey_params["mode"] == "cluster":
        pytest.skip("cluster mode requires db=0")
    in_db3 = make_valkey_adapter(valkey={"db": 3})
    _create(in_db3)
    in_db3.upsert({"id": "a", "uri": "viking://resources/a", "vector": [1, 0, 0, 0]})
    in_db0 = make_valkey_adapter(name=in_db3.collection_name)
    assert in_db3.count() == 1
    assert in_db0.collection_exists() is False
    assert in_db3._client.dbsize() == 2  # record + collection meta
