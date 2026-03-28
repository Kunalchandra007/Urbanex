"""
conftest.py — adds the project root to sys.path so all imports resolve
regardless of where pytest is invoked from.
"""
import sys
import os

# Always add the project root (directory containing this file) to the path
sys.path.insert(0, os.path.dirname(__file__))
