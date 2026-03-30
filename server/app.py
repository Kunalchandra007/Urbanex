"""
Server wrapper for URBANEX FastAPI application.

This module re-exports the FastAPI app from api.server to provide
OpenEnv-compatible module path: server.app:main
"""

from api.server import app

# Export for OpenEnv CLI: openenv validate looks for server/app.py with main() function
main = app

__all__ = ["app", "main"]
