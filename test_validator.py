#!/usr/bin/env python3
"""
Test script that simulates what the hackathon validator does.
Run this and paste the full output.
"""
import sys
import os
import json
import subprocess
from pathlib import Path

def test_inference_output():
    """Test that inference.py can run and outputs valid JSON."""
    print("\n" + "="*80)
    print("[TEST] Running inference.py (simulating validator)")
    print("="*80 + "\n")
    
    # Set SPACE_URL to point to the HF Space
    env = os.environ.copy()
    env['SPACE_URL'] = "https://kunalchandra007-urbanexx.hf.space"
    
    # Try to run inference.py
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(Path(__file__).parent)
        )
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nExit code: {result.returncode}")
        
        # Try to find and parse JSON output
        lines = result.stdout.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('['):
                json_start = i
                break
        
        if json_start >= 0:
            json_str = '\n'.join(lines[json_start:])
            try:
                data = json.loads(json_str)
                print(f"\n[OK] JSON output is valid!")
                print("Parsed JSON:")
                for task_result in data:
                    print(f"  - {task_result['task']}: score={task_result['score']}, steps={task_result['steps']}")
                return True
            except json.JSONDecodeError as e:
                print(f"\n[ERROR] JSON parsing failed: {e}")
                print("Last 500 chars of output:")
                print(result.stdout[-500:])
                return False
        else:
            print("\n[ERROR] No JSON output found in stdout!")
            return False
            
    except subprocess.TimeoutExpired:
        print("[ERROR] TIMEOUT: inference.py took longer than 120 seconds")
        return False
    except Exception as e:
        print(f"[ERROR] Error running inference.py: {e}")
        return False

def check_hf_space_up():
    """Check if HF Space is responding."""
    print("\n" + "="*80)
    print("[TEST] Checking if HF Space API is responding")
    print("="*80 + "\n")
    
    try:
        import httpx
        client = httpx.Client(timeout=10)
        response = client.get("https://kunalchandra007-urbanexx.hf.space/docs")
        if response.status_code == 200:
            print("[OK] HF Space is UP and responding")
            return True
        else:
            print(f"[WARN] HF Space returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Cannot reach HF Space: {e}")
        print("This could mean:")
        print("  - Space is still loading")
        print("  - Space is down for maintenance")
        print("  - Network connection issue")
        return False

def main():
    print("\n" + "="*80)
    print("VALIDATOR SIMULATION TEST")
    print("="*80)
    
    # Check 1: HF Space is up
    space_up = check_hf_space_up()
    
    # Check 2: Run inference.py
    inference_ok = test_inference_output()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"HF Space API up:          {'OK' if space_up else 'FAIL'}")
    print(f"inference.py output OK:    {'OK' if inference_ok else 'FAIL'}")
    
    if space_up and inference_ok:
        print("\n[OK] LIKELY TO PASS VALIDATOR")
    else:
        print("\n[ERROR] ISSUES FOUND - See details above")
    
    return 0 if (space_up and inference_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
