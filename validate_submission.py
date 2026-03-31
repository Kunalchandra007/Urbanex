#!/usr/bin/env python3
"""
Diagnostic script to validate URBANEX against OpenEnv spec.
Run this to identify validation failures.
"""
import json
import sys
import os

def check_file_exists(filename):
    """Check if critical file exists."""
    if os.path.exists(filename):
        print(f"✅ {filename} exists")
        return True
    else:
        print(f"❌ {filename} MISSING")
        return False

def check_inference_output():
    """Simulate what validator will do."""
    print("\n" + "="*60)
    print("CHECKING: inference.py output format")
    print("="*60)
    
    try:
        import inference
        # Test that functions exist
        assert hasattr(inference, 'run_inference'), "run_inference function missing"
        assert hasattr(inference, 'llm_decide_action'), "llm_decide_action function missing"
        print("✅ inference.py has required functions")
        return True
    except Exception as e:
        print(f"❌ inference.py error: {e}")
        return False

def check_openenv_yaml():
    """Validate openenv.yaml structure."""
    print("\n" + "="*60)
    print("CHECKING: openenv.yaml schema")
    print("="*60)
    
    try:
        import yaml
        with open('openenv.yaml', 'r') as f:
            spec = yaml.safe_load(f)
        
        required_keys = ['name', 'version', 'description', 'spec_version', 'action', 'observation', 'client']
        for key in required_keys:
            if key in spec:
                print(f"✅ openenv.yaml.{key} present")
            else:
                print(f"❌ openenv.yaml.{key} MISSING")
                return False
        
        # Check action/observation schemas
        if 'action' in spec and 'observation' in spec:
            print("✅ Action and observation schemas defined")
        
        return True
    except Exception as e:
        print(f"❌ openenv.yaml error: {e}")
        return False

def check_docker():
    """Check Dockerfile."""
    print("\n" + "="*60)
    print("CHECKING: Dockerfile")
    print("="*60)
    
    if not os.path.exists('Dockerfile'):
        print("❌ Dockerfile missing")
        return False
    
    with open('Dockerfile', 'r') as f:
        content = f.read()
    
    checks = [
        ('api.server:app' in content, "Entry point is api.server:app"),
        ('python:3.12' in content or 'python:3.11' in content, "Python version specified"),
        ('EXPOSE 7860' in content, "Port 7860 exposed"),
        ('requirements.txt' in content, "requirements.txt copied"),
    ]
    
    for check, desc in checks:
        if check:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} MISSING")
    
    return all(c[0] for c in checks)

def check_response_schemas():
    """Check if Pydantic models match openenv.yaml."""
    print("\n" + "="*60)
    print("CHECKING: Response schemas")
    print("="*60)
    
    try:
        from models.observation import Observation, RouteOption, Incident
        from models.action import Action
        
        # Check Observation has all required fields
        obs_fields = set(Observation.model_fields.keys())
        required_obs_fields = {
            'step', 'current_location', 'destination', 'available_routes',
            'active_incidents', 'traffic_level', 'weather', 'current_route',
            'distance_remaining_km', 'episode_done'
        }
        
        if required_obs_fields.issubset(obs_fields):
            print(f"✅ Observation has all {len(required_obs_fields)} required fields")
        else:
            missing = required_obs_fields - obs_fields
            print(f"❌ Observation missing fields: {missing}")
            return False
        
        # Check RouteOption fields
        route_fields = set(RouteOption.model_fields.keys())
        required_route_fields = {
            'route_id', 'estimated_time_min', 'incident_count',
            'fuel_cost_score', 'safety_score', 'hidden_risk_prob'
        }
        
        if required_route_fields.issubset(route_fields):
            print(f"✅ RouteOption has all {len(required_route_fields)} required fields")
        else:
            missing = required_route_fields - route_fields
            print(f"❌ RouteOption missing fields: {missing}")
            return False
        
        print("✅ All Pydantic schemas match expected structure")
        return True
        
    except Exception as e:
        print(f"❌ Schema check error: {e}")
        return False

def check_secrets():
    """Check if HF Space secrets are likely set."""
    print("\n" + "="*60)
    print("CHECKING: HF Space secrets")
    print("="*60)
    
    # These should be set in HF Space environment
    required_secrets = ['API_BASE_URL', 'MODEL_NAME', 'HF_TOKEN']
    
    for secret in required_secrets:
        if os.getenv(secret):
            print(f"✅ {secret} is set in environment")
        else:
            print(f"⚠️  {secret} not in current environment (OK if set in HF Space)")
    
    # Check inference.py reads them
    with open('inference.py', 'r') as f:
        inf_content = f.read()
    
    for secret in required_secrets:
        if f'os.getenv("{secret}"' in inf_content or f"os.getenv('{secret}'" in inf_content:
            print(f"✅ inference.py reads {secret}")
        else:
            print(f"❌ inference.py doesn't read {secret}")
    
    return True

def main():
    """Run all checks."""
    print("\n" + "="*80)
    print("🔍 URBANEX VALIDATION DIAGNOSTIC")
    print("="*80)
    
    checks_passed = 0
    checks_total = 0
    
    # File checks
    print("\n" + "="*60)
    print("CHECKING: Required files")
    print("="*60)
    files = ['Dockerfile', 'inference.py', 'openenv.yaml', 'api/server.py', 'requirements.txt']
    for f in files:
        checks_total += 1
        if check_file_exists(f):
            checks_passed += 1
    
    # Detailed checks
    checks_total += 1
    if check_inference_output():
        checks_passed += 1
    
    checks_total += 1
    if check_docker():
        checks_passed += 1
    
    checks_total += 1
    if check_response_schemas():
        checks_passed += 1
    
    check_openenv_yaml()
    check_secrets()
    
    # Summary
    print("\n" + "="*80)
    print(f"RESULT: {checks_passed}/{checks_total} major checks passed")
    print("="*80 + "\n")
    
    if checks_passed == checks_total:
        print("✅ All checks passed! Ready for submission.")
        return 0
    else:
        print(f"⚠️  {checks_total - checks_passed} checks failed. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
