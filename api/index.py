"""Vercel serverless entry point.

Vercel's Python runtime serves the ASGI ``app`` exported here. The repository
root is added to ``sys.path`` so the ``backend`` and ``core`` packages (shipped
via ``includeFiles`` in vercel.json) are importable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app  # noqa: E402

__all__ = ["app"]
