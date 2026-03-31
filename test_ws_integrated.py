#!/usr/bin/env python3
"""Test WebSocket endpoint with integrated server start."""
import subprocess
import asyncio
import json
import websockets
import time
import sys
import os

async def test_websocket():
    try:
        # Give server time to start
        await asyncio.sleep(2)
        
        async with websockets.connect('ws://localhost:7860/ws', close_timeout=5) as ws:
            print("✓ WebSocket connected")
            
            # Test 1: Reset
            print("\n1. Testing reset...")
            reset_msg = {"type": "reset", "task": "easy", "seed": 42}
            await ws.send(json.dumps(reset_msg))
            reset_resp = await ws.recv()
            reset_data = json.loads(reset_resp)
            print(f"   Response type: {reset_data.get('type')}")
            
            obs = reset_data.get('observation', {})
            required_fields = ['step', 'current_location', 'destination', 'available_routes', 
                             'active_incidents', 'traffic_level', 'weather', 'current_route', 
                             'distance_remaining_km', 'episode_done']
            missing = [f for f in required_fields if f not in obs]
            if missing:
                print(f"   ❌ MISSING FIELDS: {missing}")
                return False
            print(f"   ✓ All required fields present")
            
            # Test 2: Step
            print("\n2. Testing step...")
            step_msg = {"type": "step", "action": {"action_type": "continue"}}
            await ws.send(json.dumps(step_msg))
            step_resp = await ws.recv()
            step_data = json.loads(step_resp)
            print(f"   Reward: {step_data.get('reward')} (type: {type(step_data.get('reward')).__name__})")
            print(f"   Done: {step_data.get('done')}")
            
            if not isinstance(step_data.get('reward'), (int, float)):
                print(f"   ❌ Reward is not a number!")
                return False
            print(f"   ✓ Step response valid")
            
            print("\n✅ All tests PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Start server in background
print("Starting server...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.server:app", "--port", "7860", "--log-level", "error"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.getcwd()
)

try:
    # Run test
    success = asyncio.run(test_websocket())
    sys.exit(0 if success else 1)
finally:
    server_proc.terminate()
    server_proc.wait(timeout=5)
