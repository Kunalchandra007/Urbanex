#!/usr/bin/env python3
"""Test WebSocket endpoint locally."""
import asyncio
import json
import websockets
import sys

async def test_websocket():
    try:
        async with websockets.connect('ws://localhost:7860/ws') as ws:
            print("✓ WebSocket connected")
            
            # Test 1: Reset
            print("\n1. Testing reset...")
            reset_msg = {"type": "reset", "task": "easy", "seed": 42}
            await ws.send(json.dumps(reset_msg))
            reset_resp = await ws.recv()
            reset_data = json.loads(reset_resp)
            print(f"   Response type: {reset_data.get('type')}")
            print(f"   Episode ID: {reset_data.get('episode_id')}")
            print(f"   Obs keys: {list(reset_data.get('observation', {}).keys())}")
            
            # Verify observation has required fields
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
            print(f"   Response type: {step_data.get('type')}")
            print(f"   Reward: {step_data.get('reward')} (type: {type(step_data.get('reward')).__name__})")
            print(f"   Done: {step_data.get('done')} (type: {type(step_data.get('done')).__name__})")
            print(f"   Obs keys: {list(step_data.get('observation', {}).keys())}")
            
            # Verify step response
            if step_data.get('type') != 'step':
                print(f"   ❌ Wrong response type: {step_data.get('type')}")
                return False
            if not isinstance(step_data.get('reward'), (int, float)):
                print(f"   ❌ Reward is not a number: {type(step_data.get('reward'))}")
                return False
            if not isinstance(step_data.get('done'), bool):
                print(f"   ❌ Done is not a bool: {type(step_data.get('done'))}")
                return False
            print(f"   ✓ Step response valid")
            
            # Test 3: State
            print("\n3. Testing state...")
            state_msg = {"type": "state"}
            await ws.send(json.dumps(state_msg))
            state_resp = await ws.recv()
            state_data = json.loads(state_resp)
            print(f"   Response type: {state_data.get('type')}")
            print(f"   Obs keys: {list(state_data.get('observation', {}).keys())}")
            
            if state_data.get('type') != 'state':
                print(f"   ❌ Wrong response type: {state_data.get('type')}")
                return False
            print(f"   ✓ State response valid")
            
            print("\n✅ All WebSocket tests PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket())
    sys.exit(0 if success else 1)
