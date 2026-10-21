# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Score conversion per metric and event time decay, compared with the local engine."""

from datetime import datetime, timezone

import pytest

from openviking.storage.collection_schemas import CollectionSchemas

pytestmark = pytest.mark.integration

VECTORS = {
    "same": [0.6, 0.8, 0.0, 0.0],
    "close": [0.5, 0.7, 0.1, 0.0],
    "orthogonal": [0.0, 0.0, 0.3, 0.4],
    "opposite": [-0.6, -0.8, 0.0, 0.0],
    "small": [0.05, 0.05, 0.05, 0.05],
}
QUERY = [0.6, 0.8, 0.0, 0.0]


def _load(adapter, name, distance, records):
    adapter.create_collection(
        name,
        CollectionSchemas.context_collection(name, 4),
        distance=distance,
        sparse_weight=0.0,
        index_name="default",
    )
    adapter.upsert([dict(record) for record in records])


def _records():
    return [
        {"id": key, "uri": f"viking://resources/{key}.md", "vector": vector, "level": 2}
        for key, vector in VECTORS.items()
    ]


@pytest.mark.parametrize("distance", ["cosine", "ip", "l2"])
def test_scores_match_local_engine_scale(make_valkey_adapter, make_local_adapter, distance):
    local = make_local_adapter(name="scores")
    valkey = make_valkey_adapter(distance_metric=distance)
    _load(local, "scores", distance, _records())
    _load(valkey, valkey.collection_name, distance, _records())

    local_hits = local.query(query_vector=QUERY, limit=10)
    valkey_hits = valkey.query(query_vector=QUERY, limit=10)

    assert [hit["id"] for hit in valkey_hits] == [hit["id"] for hit in local_hits]
    # The local engine stores int8-quantized vectors (Quant: int8), so allow its error.
    for local_hit, valkey_hit in zip(local_hits, valkey_hits, strict=True):
        assert valkey_hit["_score"] == pytest.approx(local_hit["_score"], abs=0.02), local_hit["id"]


def test_event_time_decay_matches_local(make_valkey_adapter, make_local_adapter):
    records = []
    for index in range(30):
        records.append(
            {
                "id": f"e{index:02d}",
                "uri": f"viking://user/alice/memories/events/e{index:02d}.md",
                "level": 2,
                "search_tags": ["memory_type=events"],
                "updated_at": f"2026-0{1 + index % 9}-{1 + index % 27:02d}T00:00:00Z",
                "vector": [1.0, index / 30.0, (index % 4) / 4.0, 0.2],
            }
        )
    local = make_local_adapter(name="decay")
    valkey = make_valkey_adapter()
    _load(local, "decay", "cosine", records)
    _load(valkey, valkey.collection_name, "cosine", records)
    advance = {
        "time_decay": {
            "protection": "7d",
            "origin": datetime(2026, 9, 1, tzinfo=timezone.utc).isoformat(),
        }
    }
    query = [1.0, 0.3, 0.1, 0.2]

    local_hits = local.query(query_vector=query, limit=8, offset=2, advance=advance)
    valkey_hits = valkey.query(query_vector=query, limit=8, offset=2, advance=advance)

    assert [hit["id"] for hit in valkey_hits] == [hit["id"] for hit in local_hits]
    for local_hit, valkey_hit in zip(local_hits, valkey_hits, strict=True):
        for key in ("_score", "_origin_score", "_time_score"):
            assert valkey_hit[key] == pytest.approx(local_hit[key], rel=0.02, abs=1e-6), key
