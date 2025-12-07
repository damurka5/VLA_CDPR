#!/usr/bin/env python3
# check_openvla_utils.py
import sys
sys.path.append("/root/repo/openvla-oft")

print("🔍 Checking openvla_utils module...")

try:
    from experiments.robot.openvla_utils import get_vla, get_action_head, get_processor, get_proprio_projector
    print("✅ Successfully imported openvla_utils functions")
    
    import inspect
    print(f"\n📝 get_vla function:")
    print(inspect.getsource(get_vla)[:500])
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()