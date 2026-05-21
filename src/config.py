"""Path resolution for the FlyWire data directory.

The data lives outside this repository. Each environment sets the absolute
path via the ``DROSOPHILA_DATA_DIR`` environment variable, typically in
``.env.local``.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env / .env.local so DROSOPHILA_DATA_DIR is visible at import time.
try:
    from dotenv import load_dotenv

    _base_env = os.path.join(PROJECT_ROOT, ".env")
    _local_env = os.path.join(PROJECT_ROOT, ".env.local")
    if os.path.exists(_base_env):
        load_dotenv(_base_env, override=False)
    if os.path.exists(_local_env):
        load_dotenv(_local_env, override=True)
except ImportError:
    pass


DATA_DIR = os.path.abspath(
    os.path.expanduser(os.getenv("DROSOPHILA_DATA_DIR", os.path.join(PROJECT_ROOT, "data")))
)
