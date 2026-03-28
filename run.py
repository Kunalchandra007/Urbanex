"""
run.py — Start the URBANEX API server.
Run from anywhere: python run.py
"""
import sys
import os

# Ensure the project root is on sys.path for bare imports to work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=7860, reload=True)
