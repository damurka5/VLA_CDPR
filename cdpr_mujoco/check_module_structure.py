#!/usr/bin/env python3
# check_module_structure.py
import sys
sys.path.append("/root/repo/openvla-oft")

try:
    import prismatic
    print("prismatic module found")
    print("Dir:", dir(prismatic))
except Exception as e:
    print(f"Error importing prismatic: {e}")

try:
    from prismatic.vla import constants
    print("constants module found")
    print("NUM_ACTIONS_CHUNK:", constants.NUM_ACTIONS_CHUNK)
except Exception as e:
    print(f"Error importing constants: {e}")

try:
    from prismatic.models.vla import VLA
    print("VLA class found via prismatic.models.vla")
except Exception as e:
    print(f"Error importing VLA from prismatic.models.vla: {e}")
    try:
        from prismatic.vla.vla import VLA
        print("VLA class found via prismatic.vla.vla")
    except Exception as e2:
        print(f"Error importing VLA from prismatic.vla.vla: {e2}")