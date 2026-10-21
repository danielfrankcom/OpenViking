# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""TLS connection settings against a TLS-only server with a self-signed certificate."""

import os

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

pytestmark = pytest.mark.integration

TLS_URL = os.environ.get("OPENVIKING_VALKEY_TEST_TLS_URL")


def _tls_config(**valkey) -> VectorDBBackendConfig:
    if not TLS_URL:
        pytest.skip("OPENVIKING_VALKEY_TEST_TLS_URL is not configured")
    return VectorDBBackendConfig(
        backend="valkey", name="tls_test", dimension=4, valkey={"endpoints": [TLS_URL], **valkey}
    )


def test_rediss_endpoint_with_skip_verify_round_trips():
    adapter = create_collection_adapter(_tls_config(tls_insecure_skip_verify=True))
    try:
        adapter.drop_collection()
        adapter.create_collection(
            "tls_test",
            CollectionSchemas.context_collection("tls_test", 4),
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        adapter.upsert({"id": "a", "uri": "viking://resources/a", "vector": [1, 0, 0, 0]})
        assert [r["id"] for r in adapter.query(query_vector=[1, 0, 0, 0], limit=1)] == ["a"]
        adapter.drop_collection()
    finally:
        adapter.close()


def test_self_signed_certificate_is_rejected_without_skip_verify():
    with pytest.raises(RedisConnectionError, match="(?i)certificate"):
        create_collection_adapter(_tls_config())
