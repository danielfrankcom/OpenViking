# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""VectorDB backend collection adapter package."""

from .base import CollectionAdapter
from .factory import create_collection_adapter
from .http_adapter import HttpCollectionAdapter
from .local_adapter import CuVSCollectionAdapter, LocalCollectionAdapter
from .opengauss_adapter import OpenGaussCollectionAdapter
from .valkey_adapter import ValkeyCollectionAdapter
from .vikingdb_private_adapter import VikingDBPrivateCollectionAdapter
from .volcengine_adapter import VolcengineCollectionAdapter

__all__ = [
    "CollectionAdapter",
    "LocalCollectionAdapter",
    "CuVSCollectionAdapter",
    "HttpCollectionAdapter",
    "VolcengineCollectionAdapter",
    "VikingDBPrivateCollectionAdapter",
    "OpenGaussCollectionAdapter",
    "ValkeyCollectionAdapter",
    "create_collection_adapter",
]
