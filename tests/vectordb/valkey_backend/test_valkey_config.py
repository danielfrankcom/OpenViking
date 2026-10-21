# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""ValkeyConfig validation (no server needed)."""

import pytest

from openviking.storage.vectordb_adapters.valkey_adapter import ValkeyCollectionAdapter
from openviking_cli.utils.config.vectordb_config import ValkeyConfig


@pytest.mark.parametrize(
    "bad_params",
    [
        {"mode": "replicated"},
        {"mode": "sentinel"},  # requires master_name
        {"endpoints": ["127.0.0.1:6379"]},
        {"endpoints": ["redis://a:1", "redis://b:2"]},
        {"endpoints": ["redis://a:1", "rediss://b:2"], "mode": "cluster"},
        {"mode": "cluster", "db": 2},
        {"endpoints": ["redis://user:pw@127.0.0.1:6379"]},
        {"tls_insecure_skip_verify": True},  # requires rediss://
        {"index_type": "ivf"},
        {"unknown_option": 1},
        {"password": "x", "password_env": "Y"},
    ],
)
def test_invalid_config_is_rejected(bad_params):
    with pytest.raises(ValueError):
        ValkeyConfig.model_validate(bad_params)


def test_sentinel_with_tls_is_accepted():
    config = ValkeyConfig.model_validate(
        {
            "mode": "sentinel",
            "endpoints": ["rediss://10.0.0.1:26379", "rediss://10.0.0.2:26379"],
            "master_name": "vectors",
            "sentinel_password_env": "SENTINEL_PASSWORD",
            "tls_insecure_skip_verify": True,
        }
    )
    assert (config.mode, config.master_name, config.index_type) == ("sentinel", "vectors", "flat")


def test_unset_password_env_is_rejected_before_connecting(monkeypatch):
    monkeypatch.delenv("OV_VALKEY_TEST_UNSET", raising=False)
    config = ValkeyConfig(endpoints=["redis://127.0.0.1:1"], password_env="OV_VALKEY_TEST_UNSET")
    with pytest.raises(ValueError, match="password_env"):
        ValkeyCollectionAdapter("unset", "default", config)
