"""Make the training/ modules importable from the test suite."""
import os
import sys

# The app loads model heads on a background thread in production (instant bind + warming page).
# Tests need them present the moment `import predict` returns, so force the synchronous load path.
os.environ.setdefault("FLAVORMANCER_BLOCKING_LOAD", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
