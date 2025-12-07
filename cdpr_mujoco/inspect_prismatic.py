#!/usr/bin/env python3
# inspect_prismatic.py
import sys
sys.path.append("/root/repo/openvla-oft")

import os
import inspect

print("🔍 Inspecting prismatic.vla module...")

try:
    import prismatic.vla as vla_module
    print("✅ prismatic.vla imported")
    
    # List all attributes
    print("\n📋 Contents of prismatic.vla:")
    for name in dir(vla_module):
        if not name.startswith('_'):
            obj = getattr(vla_module, name)
            print(f"  - {name}: {type(obj).__name__}")
    
    # Check if there's a VLA class somewhere
    print("\n🔎 Looking for VLA class...")
    
    # Check submodules
    import pkgutil
    for _, module_name, _ in pkgutil.iter_modules(vla_module.__path__):
        print(f"  Submodule: {module_name}")
        try:
            submodule = __import__(f'prismatic.vla.{module_name}', fromlist=[''])
            for name in dir(submodule):
                if 'VLA' in name:
                    print(f"    Found {name} in prismatic.vla.{module_name}")
        except:
            pass
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Also check experiments module
print("\n🔍 Checking experiments module...")
try:
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    print("✅ GenerateConfig imported")
    
    # Check what get_vla returns
    from experiments.robot.openvla_utils import get_vla
    print("✅ get_vla imported")
    
    # Try to see what type it returns
    import inspect
    print(f"\n📝 get_vla function signature:")
    print(inspect.signature(get_vla))
    
except Exception as e:
    print(f"❌ Error importing from experiments: {e}")