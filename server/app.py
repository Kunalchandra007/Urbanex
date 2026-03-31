"""
OpenEnv-compliant FastAPI server for URBANEX.

Uses openenv.core.env_server.create_app() to automatically provision:
- WebSocket /ws endpoint (OpenEnv protocol)
- REST fallback endpoints 
- Automatic schema generation
- Health checks
"""

from openenv.core.env_server import create_app
from models import UrbanexAction, UrbanexObservation
from server.urbanex_environment import UrbanexEnvironment

# Create the FastAPI app using OpenEnv's factory
# This automatically sets up:
# - POST /reset, /step for REST API
# - WebSocket /ws for OpenEnv protocol
# - GET /metadata, /schema for introspection
# - GET /health for readiness checks
app = create_app(
    env_class=UrbanexEnvironment,
    action_type=UrbanexAction,
    observation_type=UrbanexObservation,
    env_name="urbanex",
)


def main():
    """Entry point for server startup."""
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]

