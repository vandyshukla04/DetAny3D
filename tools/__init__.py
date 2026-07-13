"""DetAny3D tooling.

Marking this a regular package is load-bearing: an unrelated `tools` distribution
exists in site-packages, and Python resolves regular packages BEFORE namespace
packages, so without this file `import tools.heading` silently binds to the
site-packages one instead of this repo's. Scripts under tools/ are still runnable
directly (`python tools/foo.py`); this only makes `import tools.x` deterministic.
"""
