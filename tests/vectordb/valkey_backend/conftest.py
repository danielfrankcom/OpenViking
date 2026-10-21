# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fixtures for the Valkey Search backend tests.

Tests that need a server are skipped unless OPENVIKING_VALKEY_TEST_URL is set, e.g.
``redis://127.0.0.1:6379``. The server must run Valkey Search >= 1.2.1 with
``search.max-vector-knn 100000``. Optional:

* OPENVIKING_VALKEY_TEST_MODE: standalone (default), cluster or sentinel. In sentinel
  mode the URL is a Sentinel and OPENVIKING_VALKEY_TEST_MASTER_NAME names the primary
  (default ``mymaster``).
* OPENVIKING_VALKEY_TEST_TLS_URL: a TLS-only server with a self-signed certificate.
"""

import os
import uuid

import pytest

from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig


def _server_params() -> dict:
    url = os.environ.get("OPENVIKING_VALKEY_TEST_URL")
    if not url:
        pytest.skip("OPENVIKING_VALKEY_TEST_URL is not configured")
    params = {
        "endpoints": [url],
        "mode": os.environ.get("OPENVIKING_VALKEY_TEST_MODE", "standalone"),
    }
    if params["mode"] == "sentinel":
        params["master_name"] = os.environ.get("OPENVIKING_VALKEY_TEST_MASTER_NAME", "mymaster")
    return params


@pytest.fixture
def valkey_params() -> dict:
    """The ``valkey`` config section for the test server."""
    return _server_params()


@pytest.fixture(scope="session")
def valkey_config():
    """Build a ``backend: valkey`` config for the test server; ``valkey`` overrides fields."""

    def build(name: str, dimension: int, valkey: dict | None = None, **kwargs):
        return VectorDBBackendConfig(
            backend="valkey",
            name=name,
            dimension=dimension,
            valkey={**_server_params(), **(valkey or {})},
            **kwargs,
        )

    return build


@pytest.fixture
def make_valkey_adapter(valkey_config):
    """Create adapters on fresh collection names; drop and close them afterwards."""
    adapters = []

    def make(dimension: int = 4, name: str | None = None, **kwargs):
        name = name or f"t{uuid.uuid4().hex[:12]}"
        adapter = create_collection_adapter(valkey_config(name, dimension, **kwargs))
        adapters.append(adapter)
        return adapter

    yield make
    dropped = set()
    for adapter in adapters:
        # Adapters sharing a name cache the same collection; drop it only once.
        if adapter.collection_name not in dropped:
            adapter.drop_collection()
            dropped.add(adapter.collection_name)
        adapter.close()


@pytest.fixture
def make_local_adapter(tmp_path):
    adapters = []

    def make(dimension: int = 4, name: str = "context"):
        config = VectorDBBackendConfig(
            backend="local",
            name=name,
            dimension=dimension,
            path=str(tmp_path / f"local-{len(adapters)}"),
        )
        adapter = create_collection_adapter(config)
        adapters.append(adapter)
        return adapter

    yield make
    for adapter in adapters:
        adapter.close()
