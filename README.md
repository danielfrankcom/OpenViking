# OpenViking: Valkey Search vector backend (prototype)

This branch is a **prototype** of the `valkey` vector backend proposed in [volcengine/OpenViking#6492](https://github.com/volcengine/OpenViking/issues/6492). It isn't part of upstream OpenViking. It's based on upstream `main` at `4ab22a4`, and this README replaces upstream's on this branch only. For OpenViking itself, see the [upstream repository](https://github.com/volcengine/OpenViking).

## Changes on this branch

- A `valkey` backend that stores OpenViking's vectors in [Valkey Search](https://github.com/valkey-io/valkey-search), registered next to the existing backends. It fits the existing adapter interface without changing it or its callers.
- A `storage.vectordb.valkey` configuration section that reuses the RAGFS cache's Redis connection settings. It supports standalone, cluster, and Sentinel deployments, with TLS.
- An optional `valkey` dependency extra, which installs `redis-py`.
- Tests under `tests/vectordb/valkey_backend/`, and a CI job that runs them against a Valkey container.

## Requirements

- Valkey Search 1.2.1 or later, on Valkey 9.0.6 or 9.1.2 and later. `valkey/valkey-bundle:9.1.3` meets both.
- `search.max-vector-knn` set to 100000, because OpenViking's fetch-all queries request up to 100,000 records.
- `search.max-nonvector-search-results-fetched` only needs raising if one node stores more than 100,000 records. Above its limit, the backend raises an error instead of returning truncated results.

## Configuration

```json
{
  "storage": {
    "vectordb": {
      "backend": "valkey",
      "distance_metric": "cosine",
      "valkey": {
        "mode": "standalone",
        "endpoints": ["redis://127.0.0.1:6379"]
      }
    }
  }
}
```

`mode` can be `standalone`, `cluster`, or `sentinel`, with the same fields as the RAGFS cache's Redis settings. The backend adds `index_type`, which is `flat` (exact, the default) or `hnsw`, and `vector_index_params`, for extra HNSW settings.

## Running the tests

Start Valkey with the required setting:

```bash
docker run -d --name valkey -p 6379:6379 \
  -e VALKEY_EXTRA_FLAGS="--search.max-vector-knn 100000" \
  valkey/valkey-bundle:9.1.3
```

Then install, build the C++ engine, and run the backend tests. The differential tests compare results with the local engine, so they need the C++ extension.

```bash
uv sync --extra test --extra valkey
uv pip install setuptools setuptools_scm cmake wheel
uv run python setup.py build_ext --inplace
OPENVIKING_VALKEY_TEST_URL=redis://127.0.0.1:6379 \
  uv run pytest -q -o addopts='' tests/vectordb/valkey_backend
```

On macOS the C++ build may need `CXXFLAGS=-Wno-error=#warnings CFLAGS=-Wno-error=#warnings`.

| Variable | Purpose |
| --- | --- |
| `OPENVIKING_VALKEY_TEST_URL` | Server for the live tests. Without it, they're skipped. |
| `OPENVIKING_VALKEY_TEST_MODE` | `standalone` (default), `cluster`, or `sentinel` |
| `OPENVIKING_VALKEY_TEST_MASTER_NAME` | Sentinel master name (default `mymaster`) |
| `OPENVIKING_VALKEY_TEST_TLS_URL` | A `rediss://` server for the TLS tests |

## Validation so far

On Valkey Search 1.2.1 and 1.3.0-rc1, in standalone, cluster, and Sentinel deployments:

- The backend suite passes (181 tests in standalone and Sentinel, 180 in cluster, with the rest skipped for that mode).
- OpenViking's storage tests give the same per-test results as on the local backend.
- Filters, including every directory-scope depth, `level`, account scope, multi-value tags, and `contains`/`regex`, return the same records as the local engine. Cosine, IP, and L2 scores match it too.
- On earlier commits of this branch, `find` and `search` through a running server matched the local backend before and after reindex, move, and delete, and a 40-minute mixed workload on a cluster had no search or visibility failures.
- Servers that start concurrently against an empty deployment all come up working.

## Known issues

- **Dense vectors only.** Like the openGauss backend, it rejects `sparse_weight > 0`. Sparse query vectors are ignored when `sparse_weight` is 0.
- **`contains` and `regex` are evaluated in the backend.** Valkey Search has no substring match. A query with these clauses reads every record that matches the rest of its filter. On a 50,000-record collection, such a query took up to about 3.6 s. A grouped count combined with them is rejected, and `geo_range` is rejected.
- **Truncation detection depends on `CONFIG GET`.** If the application's Valkey user can't run it, the backend assumes the default result cap of 100000. The backend wouldn't detect a lower cap.
- **Many indexes on a cluster.** In cluster mode, creating an index gets slower as the number of indexes grows, so a cluster shouldn't hold hundreds of them. OpenViking's broad test selection leaves that many behind, but a normal deployment uses one index.
- **Records without a sort field.** In sorted results, this backend keeps records that don't have the sort field and places them last. The local engine handles these inconsistently, dropping them when there's no filter and misplacing them when there is one.
- **OpenViking features that don't support this backend**, as with the openGauss backend: account-owned vector databases, the distance-metric check, and automatic schema updates. `ValkeyConfig` also isn't exported from the config package.
- **`uv.lock`** also locks `numpy` for the `bot` extra, because upstream `main`'s lock file is out of date for it. That commit can be dropped once upstream fixes it.
- **One unexplained failure.** In one Sentinel run, `clear()` left 34 records behind. It didn't happen again in many later runs.

## Still to check before an upstream PR

- Re-run the broad OpenViking test selection and the full-stack comparison on the rebased branch.
- Detect a lowered result cap even when the application's Valkey user can't run `CONFIG GET`, such as through a configuration field for the cap.
- Work out and document the minimal ACL for the application's Valkey user.
- Test replicas and failover.
- Test at 100,000 records and beyond, and compare latency with the local backend, including HNSW recall.
- Run the CI job on GitHub. Check that the service container receives `VALKEY_EXTRA_FLAGS`, and how long the C++ build takes.
- Write the English and Chinese docs: backend table, configuration, requirements, and limitations.
- Settle the design points a reviewer is likely to question:
  - `ValkeyConfig` extends the RAGFS cache's Redis config, so it inherits `default_ttl_seconds`, which doesn't apply to vectors.
  - The backend sets up its own synchronous client instead of reusing the account store's asynchronous one.
  - The adapter pings the server when it's created.
  - `redis-py` raises an error when its connection pool is full.
  - Some tests read the backend's private state.
- Resolve the two `Open decision` comments in the code: the minimum version and the missing-sort-field behavior.
