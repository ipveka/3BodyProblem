"""Make the project packages importable when running the tests in-place."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
