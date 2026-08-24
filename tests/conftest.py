"""Session-wide pytest setup.

Sets two environment variables *before* anything else in the test session gets a
chance to import `torch` or `lightgbm` -- both bundle their own `libomp` runtime on
macOS, and mixing them in one process is unsafe. `conftest.py` is the first thing
pytest imports, before collecting any test module, so this is early enough to take
effect.

`OMP_NUM_THREADS=1` avoids each library spinning up its own competing thread pool;
`KMP_DUPLICATE_LIB_OK=TRUE` is Intel OpenMP's own escape hatch for "yes, I know two
copies of libomp are loaded, proceed anyway" (the two libraries are otherwise mutually
exclusive in one process by design, precisely to prevent silent data races).

**These two env vars reduce the failure rate but do not fully fix it** -- confirmed by
reproducing a `Fatal Python error: Segmentation fault` inside LightGBM's Dataset
construction (`lightgbm/basic.py:__init_from_np2d`) with both vars set and
`lightgbm_baseline`'s own `num_threads=1` pinned, running the full suite. The actual
fix is process-level, not thread-level: the Makefile's `test` target and
`.github/workflows/ci.yml` both run the PyTorch tests
(`test_mlp_entity_embeddings.py`, `test_gru_forecast.py`) in a **separate pytest
process** from every LightGBM test, so the two libraries are never resident in the
same process at all. `pytest -p no:cacheprovider` / a bare `pytest` invocation that
collects both groups together will still intermittently hang or crash -- always use
`make test` (or the equivalent two-invocation command) instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
