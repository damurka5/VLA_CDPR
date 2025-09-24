import os
# Set the paths explicitly
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/root/.mujoco/mujoco200'
os.environ['LD_LIBRARY_PATH'] = '/root/.mujoco/mujoco200/bin:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    from mujoco_py import load_model_from_xml, MjSim
    print("✓ mujoco-py imported successfully!")
    
    # Test with a simple XML
    xml = """
    <mujoco>
        <worldbody>
            <light name="top" pos="0 0 1"/>
            <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>
            <body name="box" pos="0 0 0.1">
                <joint type="free"/>
                <geom name="box_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.3 0.3 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    model = load_model_from_xml(xml)
    sim = MjSim(model)
    sim.step()
    print("Basic simulation test passed!")
    
except Exception as e:
    print(f" Error: {e}")
    import traceback
    traceback.print_exc()