#!/usr/bin/env python3
"""
Test script that simulates the hackathon validator's inference parsing.
Run this and paste the full output.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

START_RE = re.compile(r"^\[START\] task=(\w+) env=urbanex model=(.+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(\d+) action=([^\s]+) reward=(-?\d+\.\d{2}) "
    r"done=(true|false) error=(.+)$"
)
END_RE = re.compile(
    r"^\[END\] task=(\w+) success=(true|false) score=(\d+\.\d{2}) "
    r"steps=(\d+) rewards=(.*)$"
)


def test_inference_output():
    """Test that inference.py emits validator-compatible trace lines."""
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
        
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        current_task = None
        ended_tasks = []
        step_counts = {}

        for line in lines:
            start_match = START_RE.match(line)
            if start_match:
                current_task = start_match.group(1)
                step_counts[current_task] = 0
                continue

            step_match = STEP_RE.match(line)
            if step_match:
                if current_task is None:
                    print(f"\n[ERROR] STEP appeared before START: {line}")
                    return False
                reward = float(step_match.group(3))
                if not (-0.95 <= reward <= 0.95):
                    print(f"\n[ERROR] STEP reward out of safe display range: {line}")
                    return False
                step_counts[current_task] += 1
                continue

            end_match = END_RE.match(line)
            if end_match:
                task = end_match.group(1)
                score = float(end_match.group(3))
                steps = int(end_match.group(4))
                rewards = [part for part in end_match.group(5).split(",") if part]

                if not (0.0 < score < 1.0):
                    print(f"\n[ERROR] END score out of open interval: {line}")
                    return False
                if steps != step_counts.get(task, 0):
                    print(
                        f"\n[ERROR] END step count mismatch for {task}: "
                        f"expected {step_counts.get(task, 0)}, got {steps}"
                    )
                    return False
                if rewards and len(rewards) != steps:
                    print(
                        f"\n[ERROR] END rewards length mismatch for {task}: "
                        f"{len(rewards)} rewards for {steps} steps"
                    )
                    return False
                ended_tasks.append(task)
                current_task = None
                continue

        if sorted(ended_tasks) == ["easy", "hard", "medium"]:
            print("\n[OK] Inference output matches validator trace format")
            for task in ended_tasks:
                print(f"  - {task}: steps={step_counts.get(task, 0)}")
            return True

        print(f"\n[ERROR] Missing task END blocks. Saw: {ended_tasks}")
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
