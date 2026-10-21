# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Valkey Search vector database adapter using redis-py.

Valkey Search provides FLAT and HNSW vector indexes over HASH keys, with TAG and
NUMERIC prefilters. Standalone, cluster (with the Valkey Search coordinator) and
Sentinel deployments are supported.

Driver: redis-py.
  Install: pip install "openviking[valkey]"
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from urllib.parse import urlparse

from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb_adapters.base import CollectionAdapter
from openviking_cli.utils.config.vectordb_config import (
    DEFAULT_INDEX_NAME,
    ValkeyConfig,
    VectorDBBackendConfig,
)

from .valkey.collection import ValkeyCollection

# Open decision (minimum version): the backend needs Valkey Search 1.2.1 (Valkey
# 9.0.1 or later). 1.2.0 can drop replies to pipelined HSETs and older versions cannot
# escape "}" in TAG queries or run OpenViking's ACL filters. The version is not
# checked at runtime.


def _secret(value: str, env_name: str, field_name: str) -> str | None:
    if not env_name:
        return value or None
    resolved = os.environ.get(env_name)
    if not resolved:
        raise ValueError(f"valkey.{field_name}_env points to an unset or empty variable")
    return resolved


def _connect(config: ValkeyConfig) -> tuple[Any, list[Any]]:
    """Return a synchronous redis-py client and the Sentinel clients it depends on.

    The connection fields map to redis-py options the same way as the account store's
    ``create_redis_client``. That factory builds asyncio clients that decode replies;
    ICollection is synchronous and reads binary vectors, so this one does neither.
    """
    # redis-py is an optional dependency, needed only when this backend is selected.
    try:
        from redis import Redis
        from redis.cluster import ClusterNode, RedisCluster
        from redis.sentinel import Sentinel
    except ImportError as e:
        raise ImportError(
            'redis is required for the Valkey backend. Install it via: pip install "openviking[valkey]"'
        ) from e

    tls: Dict[str, Any] = {}
    if urlparse(config.endpoints[0]).scheme == "rediss":
        tls = {
            "ssl": True,
            "ssl_cert_reqs": "none" if config.tls_insecure_skip_verify else "required",
        }
    timeouts = {
        "socket_connect_timeout": config.connect_timeout_ms / 1000,
        "socket_timeout": config.command_timeout_ms / 1000,
    }
    options = {
        **timeouts,
        **tls,
        "username": config.username or None,
        "password": _secret(config.password, config.password_env, "password"),
        "max_connections": config.pool_size,
    }
    # RedisCacheConfig has checked that every endpoint is a redis(s):// URL with a host.
    addresses = [(url.hostname or "", url.port or 6379) for url in map(urlparse, config.endpoints)]

    if config.mode == "cluster":
        nodes = [ClusterNode(host, port) for host, port in addresses]
        return RedisCluster(startup_nodes=nodes, **options), []

    if config.mode == "sentinel":
        sentinel_password = _secret(
            config.sentinel_password, config.sentinel_password_env, "sentinel_password"
        )
        sentinel = Sentinel(
            addresses,
            sentinel_kwargs={
                **timeouts,
                **tls,
                "username": config.sentinel_username or None,
                "password": sentinel_password,
            },
        )
        primary = sentinel.master_for(config.master_name, db=config.db, **options)
        return primary, list(sentinel.sentinels)

    host, port = addresses[0]
    return Redis(host=host, port=port, db=config.db, **options), []


class ValkeyCollectionAdapter(CollectionAdapter):
    """Adapter for one OpenViking collection stored in Valkey Search."""

    mode = "valkey"

    def __init__(self, collection_name: str, index_name: str, config: ValkeyConfig):
        super().__init__(collection_name=collection_name, index_name=index_name)
        self._config = config
        self._client, self._sentinel_clients = _connect(config)
        # redis-py connects lazily; fail here on a wrong address, credential or
        # certificate instead of on the first vector operation.
        self._client.ping()

    @classmethod
    def from_config(cls, config: VectorDBBackendConfig) -> "ValkeyCollectionAdapter":
        if config.valkey is None:
            raise ValueError("VectorDB valkey backend requires 'valkey' config")
        return cls(
            collection_name=config.name or "context",
            index_name=config.index_name or DEFAULT_INDEX_NAME,
            config=config.valkey,
        )

    @property
    def _is_cluster(self) -> bool:
        return self._config.mode == "cluster"

    def _load_existing_collection_if_needed(self) -> None:
        if self._collection is not None:
            return
        collection = ValkeyCollection.open(self._client, self._is_cluster, self._collection_name)
        if collection is not None:
            self._collection = Collection(collection)

    def _create_backend_collection(self, meta: Dict[str, Any]) -> Collection:
        return Collection(
            ValkeyCollection.create(self._client, self._is_cluster, self._collection_name, meta)
        )

    def _build_default_index_meta(
        self,
        *,
        index_name: str,
        distance: str,
        use_sparse: bool,
        sparse_weight: float,
        scalar_index_fields: list[str],
    ) -> Dict[str, Any]:
        if use_sparse:
            raise NotImplementedError("Valkey vector backend does not support sparse vectors")
        return {
            "IndexName": index_name,
            "VectorIndex": {
                "IndexType": self._config.index_type,
                "Distance": distance,
                "Params": dict(self._config.vector_index_params),
            },
            "ScalarIndex": scalar_index_fields,
        }

    def update_data(self, data_list: List[Dict[str, Any]]) -> List[str]:
        records = [self._normalize_record_for_write(record) for record in data_list]
        return [str(record_id) for record_id in self.get_collection().update_data(records).ids]

    def close(self) -> None:
        super().close()
        self._client.close()
        for sentinel_client in self._sentinel_clients:
            sentinel_client.close()
