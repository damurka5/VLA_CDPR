#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CDPR + LIBERO scene/object adapter.
- Chooses a LIBERO scene (e.g., desk) and places LIBERO objects with given poses.
- Generates a temporary wrapper MJCF that includes: [scene] + [cdpr.xml] + [placed objects].
- Boots existing HeadlessCDPRSimulation on that wrapper.

Usage examples:
  python cdpr_scene_switcher.py --scene desk
  python cdpr_scene_switcher.py --scene desk \
      --object orange_juice:0.50,0.50,0.00:0,0,0,1 \
      --object orange_juice:0.70,0.35,0.00:0,0,0.382683,0.92388

Notes:
- Object pose format: name: x,y,z : qx,qy,qz,qw   (quat optional; defaults to 0,0,0,1)
- Scene names and object names correspond to directory names under LIBERO assets.
"""

import os, sys, argparse, tempfile, shutil, textwrap
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

# --- repo-local import of simulation class ---
sys.path.append(str(Path(__file__).parent))  # add VLA_CDPR/mujoco to path
from headless_cdpr_egl import HeadlessCDPRSimulation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # repo/
LIBERO_ASSETS = REPO / "LIBERO" / "libero" / "libero" / "assets"
SCENES_DIR = LIBERO_ASSETS / "scenes"
OBJECTS_DIR = LIBERO_ASSETS / "stable_hope_objects"

CDPR_XML = HERE / "cdpr.xml"

# replace your existing preprocess_scene_with_zoffset(...) with this:
def preprocess_scene_with_zoffset(scene_xml: Path, z_offset: float, out_xml: Path):
    """
    Load the LIBERO scene MJCF and:
      1) rewrite all <asset> file paths (mesh/texture/skin) to ABSOLUTE paths
         based on the scene directory,
      2) add z_offset (meters) to the 'pos' z of every top-level body in <worldbody>.
    Write to out_xml.
    """
    src_dir = scene_xml.parent.resolve()
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    # 1) rewrite asset file paths to absolute
    for a in root.findall("asset"):
        for sub in list(a):
            if sub.tag in ("mesh", "texture", "skin") and "file" in sub.attrib:
                rel = sub.get("file")
                abs_path = (src_dir / rel).resolve()
                sub.set("file", str(abs_path))

    # 2) shift top-level bodies' z
    for wb in root.findall("worldbody"):
        for body in list(wb.findall("body")):
            pos_str = body.get("pos", "0 0 0").split()
            if len(pos_str) == 3:
                x, y, z = map(float, pos_str)
                body.set("pos", f"{x} {y} {z + z_offset}")

    try:
        ET.indent(tree)
    except Exception:
        pass
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

def preprocess_cdpr_set_ee_start(cdpr_xml: Path, ee_xyz: np.ndarray, out_xml: Path):
    """
    Overwrite 'pos' of body name='ee_base' in cdpr.xml.
    """
    tree = ET.parse(cdpr_xml)
    root = tree.getroot()
    found = False
    for wb in root.findall("worldbody"):
        # search recursively
        stack = list(wb.findall("body"))
        while stack:
            b = stack.pop()
            if b.get("name") == "ee_base":
                b.set("pos", f"{ee_xyz[0]} {ee_xyz[1]} {ee_xyz[2]}")
                found = True
                break
            stack.extend(list(b.findall("body")))
    if not found:
        raise ValueError("Could not find body name='ee_base' in cdpr.xml.")
    try:
        ET.indent(tree)
    except Exception:
        pass
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def parse_object_arg(arg: str):
    """
    Parse: name: x,y,z : qx,qy,qz,qw
    Quat is optional; defaults to identity.
    """
    parts = [p.strip() for p in arg.split(":")]
    if len(parts) < 2:
        raise ValueError(f"Bad --object spec '{arg}'. Expected 'name:x,y,z[:qx,qy,qz,qw]'")
    name = parts[0]
    pos = np.fromstring(parts[1], sep=",", dtype=float)
    if pos.size != 3:
        raise ValueError(f"Bad position in '{arg}'. Need x,y,z")
    if len(parts) >= 3:
        quat = np.fromstring(parts[2], sep=",", dtype=float)
        if quat.size != 4:
            raise ValueError(f"Bad quaternion in '{arg}'. Need qx,qy,qz,qw")
    else:
        quat = np.array([0,0,0,1.0], dtype=float)
    return name, pos, quat

def find_scene_xml(scene_name: str) -> Path:
    cand = SCENES_DIR / scene_name / f"{scene_name}.xml"
    if not cand.exists():
        raise FileNotFoundError(f"Scene '{scene_name}' not found at {cand}")
    return cand

def find_object_xml(object_name: str) -> Path:
    # many objects live under: stable_hope_objects/<name>/<name>.xml
    cand = OBJECTS_DIR / object_name / f"{object_name}.xml"
    if cand.exists():
        return cand
    # fallback: first xml under dir
    d = OBJECTS_DIR / object_name
    xmls = list(d.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"Object '{object_name}' has no .xml in {d}")
    return xmls[0]

def make_placed_object_xml(orig_object_xml: Path, out_xml: Path, prefix: str, pos, quat, force_dynamic=False):
    import xml.etree.ElementTree as ET

    def clone(elem):
        return ET.fromstring(ET.tostring(elem))

    src_dir = orig_object_xml.parent.resolve()
    tree = ET.parse(orig_object_xml)
    root = tree.getroot()

    # Copy <asset> and absolutize mesh/texture paths (unchanged idea)
    asset_elems = []
    for a in root.findall("asset"):
        a_copy = clone(a)
        for sub in a_copy:
            if sub.tag in ("mesh", "texture", "skin") and "file" in sub.attrib:
                sub.set("file", str((src_dir / sub.get("file")).resolve()))
        asset_elems.append(a_copy)

    # Take first body under any worldbody
    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError(f"{orig_object_xml} has no <worldbody> section.")
    bodies = []
    for wb in worldbodies:
        bodies.extend(list(wb.findall("body")))
    if not bodies:
        raise ValueError(f"{orig_object_xml} has no <body> inside <worldbody>.")
    body_clone = clone(bodies[0])

    # Prefix names to avoid collisions
    NAME_TAGS = {"body", "geom", "site", "joint", "camera", "light"}
    def prefix_names(elem):
        if elem.tag in NAME_TAGS and "name" in elem.attrib:
            elem.set("name", f"{prefix}_{elem.get('name')}")
        for child in list(elem):
            prefix_names(child)
    prefix_names(body_clone)

    # Apply pose (MuJoCo wants w x y z)
    body_clone.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    wxyz = (float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
    body_clone.set("quat", f"{wxyz[0]} {wxyz[1]} {wxyz[2]} {wxyz[3]}")

    # ✅ If requested, ensure this object is dynamic:
    #    - add a <freejoint/> if there is no joint anywhere under this body
    has_joint = any(elem.tag == "joint" for elem in body_clone.iter())
    if force_dynamic and not has_joint:
        ET.SubElement(body_clone, "freejoint")

    # Build minimal MJCF; also give default density so geoms get mass if needed
    mj = ET.Element("mujoco")
    comp = ET.SubElement(mj, "compiler"); comp.set("autolimits", "true")

    default = ET.SubElement(mj, "default")
    geomdef = ET.SubElement(default, "geom")
    geomdef.set("density", "1200")  # ~plastic; adjust if you like

    if asset_elems:
        new_asset = ET.SubElement(mj, "asset")
        for a in asset_elems:
            for sub in list(a):
                new_asset.append(sub)

    new_wb = ET.SubElement(mj, "worldbody")
    new_wb.append(body_clone)

    try: ET.indent(mj)
    except: pass
    ET.ElementTree(mj).write(out_xml, encoding="utf-8", xml_declaration=True)

def build_wrapper_mjcf(scene_xml: Path, cdpr_xml: Path, placed_object_xmls: list[Path], out_xml: Path):
    scene_xml = scene_xml.resolve()
    cdpr_xml  = cdpr_xml.resolve()
    scene_dir = scene_xml.parent.resolve()
    # includes for placed objects use absolute asset paths, so no meshdir needed
    includes_objects = os.linesep.join([f'<include file="{str(p.resolve())}"/>' for p in placed_object_xmls])

    content = f"""<mujoco>
    <compiler autolimits="true"/>

    <!-- Set mesh/texture dirs for the SCENE only -->
    <compiler meshdir="{str(scene_dir)}" texturedir="{str(scene_dir)}"/>
    <include file="{str(scene_xml)}"/>

    <!-- Reset to neutral (our CDPR uses primitives / absolute includes) -->
    <compiler meshdir="" texturedir=""/>

    <!-- CDPR rig -->
    <include file="{str(cdpr_xml)}"/>

    <!-- Placed LIBERO objects (assets already absolute) -->
    {includes_objects}
    </mujoco>
    """
    out_xml.parent.mkdir(parents=True, exist_ok=True)  # ensure /.../wrappers/ exists
    out_xml.write_text(content, encoding="utf-8")


    

def main():
    ap = argparse.ArgumentParser(description="Run CDPR in a LIBERO scene with placed objects.")
    ap.add_argument("--scene", required=True, help="Scene name under assets/scenes (e.g., 'desk').")
    ap.add_argument("--object", action="append", default=[],
                    help="Object placement: name:x,y,z[:qx,qy,qz,qw]. Repeat for multiple.")
    ap.add_argument("--outdir", default=str(HERE / "trajectory_results"), help="Output video/data dir.")
    ap.add_argument("--hover", default="0.5,0.5,0.35", help="Hover xyz over object for the demo (x,y,z).")
    ap.add_argument("--graspz", type=float, default=0.06, help="Grasp height z.")
    ap.add_argument("--liftz",  type=float, default=0.35, help="Lift height z.")
    ap.add_argument("--yaw",    type=float, default=0.0,  help="Yaw target (rad) for the demo.")
    ap.add_argument("--steps",  type=int,   default=120,  help="Goto() max steps for segments.")
    ap.add_argument("--scene_z", type=float, default=0.0,
                help="Additive z-offset (meters) applied to EVERY body in the LIBERO scene (e.g., -0.25 lowers the table).")
    ap.add_argument("--ee_start", default=None,
                    help="Override CDPR ee_base start pos as 'x,y,z'. If unset, use whatever is in cdpr.xml.")
    ap.add_argument("--object_on_table", action="store_true",
                    help="Force object z to table_z.")
    ap.add_argument("--table_z", type=float, default=0.0,
                    help="Tabletop height (meters) used when --object_on_table is set.")
    ap.add_argument("--object_dynamic", action="store_true",
                help="Force placed objects to be dynamic (inject <freejoint/> if missing).")
    ap.add_argument("--settle_time", type=float, default=1.0,
                help="Seconds to simulate before the demo to let objects fall/settle.")
    ap.add_argument("--wrapper_out", default=None,
                help="Write final wrapper MJCF to this path (kept). If unset, uses a temp dir.")
    ap.add_argument("--keep", action="store_true",
                help="Don’t delete the temp working directory (for debugging).")

    args = ap.parse_args()

    # Where to write the final wrapper
    if args.wrapper_out:
        wrapper_xml = Path(args.wrapper_out).resolve()
        wrapper_xml.parent.mkdir(parents=True, exist_ok=True)
        gen_base = wrapper_xml.parent     # <- persist placed_* and overrides here
    else:
        wrapper_xml = tmpdir / "cdpr_scene_wrapper.xml"
        gen_base = tmpdir                 # <- ephemeral if no --wrapper_out


    scene_xml = find_scene_xml(args.scene)
    if not CDPR_XML.exists():
        raise FileNotFoundError(f"CDPR XML not found at {CDPR_XML}")

    # temp workspace
    tmpdir = Path(tempfile.mkdtemp(prefix="cdpr_scene_", dir=str(HERE)))

    # Scene (z-shift if needed)
    if abs(args.scene_z) > 1e-6:
        scene_for_include = gen_base / f"{args.scene}_zshift.xml"
        preprocess_scene_with_zoffset(scene_xml, args.scene_z, scene_for_include)
    else:
        scene_for_include = scene_xml

    # CDPR (override ee_base pos if requested)
    if args.ee_start is not None:
        ee_xyz = np.fromstring(args.ee_start, sep=",", dtype=float)
        if ee_xyz.size != 3:
            raise ValueError("--ee_start must be 'x,y,z'")
        cdpr_for_include = gen_base / "cdpr_ee_override.xml"
        preprocess_cdpr_set_ee_start(CDPR_XML, ee_xyz, cdpr_for_include)
    else:
        cdpr_for_include = CDPR_XML


    # Parse objects
    placements = []
    for ob in args.object:
        name, pos, quat = parse_object_arg(ob)
        if args.object_on_table:
            pos[2] = float(args.table_z)  # snap to table height
        obj_xml = find_object_xml(name)
        placements.append((name, obj_xml, pos, quat))

    # temp workspace for generated files
    tmpdir = Path(tempfile.mkdtemp(prefix="cdpr_scene_", dir=str(HERE)))
    try:
        placed_xmls = []
        for idx, (name, obj_xml, pos, quat) in enumerate(placements):
            placed_path = gen_base / f"placed_{idx}_{name}.xml"
            make_placed_object_xml(obj_xml, placed_path, prefix=f"p{idx}", pos=pos, quat=quat,
                                force_dynamic=args.object_dynamic)
            placed_xmls.append(placed_path)


        wrapper_xml = (Path(args.wrapper_out).resolve()
               if args.wrapper_out else (tmpdir / "cdpr_scene_wrapper.xml"))
        build_wrapper_mjcf(scene_for_include, cdpr_for_include, placed_xmls, wrapper_xml)

        print(f"✅ Built wrapper: {wrapper_xml}")
        print(f"   Includes {len(placed_xmls)} object(s).")

        # Only delete tmpdir if we used it for everything
        if not args.wrapper_out:
            shutil.rmtree(tmpdir, ignore_errors=True)


        # Run your existing headless sim on the wrapper
        sim = HeadlessCDPRSimulation(str(wrapper_xml), output_dir=args.outdir)
        sim.initialize()
        
        # settle objects so they drop onto the table
        if args.settle_time > 0:
            steps = max(1, int(round(args.settle_time / sim.controller.dt)))
            print(f"⏳ settling objects for {args.settle_time:.2f}s ({steps} steps)")
            for k in range(steps):
                # keep cables neutral (we're not commanding sliders here)
                sim.run_simulation_step(capture_frame=False)

        # A tiny smoke test: move to hover over the first object (if provided), do a quick yaw & squeeze
        if placements:
            ox, oy, _ = placements[0][2]
        else:
            # default hover parsed from CLI
            ox, oy, hz = np.fromstring(args.hover, sep=",", dtype=float)
        hover = np.fromstring(args.hover, sep=",", dtype=float)
        if hover.size != 3:
            hover = np.array([ox, oy, 0.35], dtype=float)

        # Use your demo (if you added yaw helpers), fallback to simple hover motion
        try:
            ok, _ = sim.goto(hover, max_steps=args.steps)
            sim.set_yaw(args.yaw)
            sim.open_gripper()
            for _ in range(40): sim.run_simulation_step(capture_frame=True)
            sim.close_gripper()
            for _ in range(40): sim.run_simulation_step(capture_frame=True)
        except Exception as e:
            print("Warning during smoke motion:", e)

        # Save a short clip and data
        # ts = "scene_switcher"
        # sim.save_trajectory_results(HERE / "trajectory_results" / ts, ts)
        
        ts = "scene_switcher"
        ts_dir = Path(HERE / "trajectory_results" / ts)
        ts_dir.mkdir(parents=True, exist_ok=True)
        sim.save_trajectory_results(str(ts_dir), ts)
        sim.cleanup()
        print(f"✅ Loaded scene '{args.scene}' with {len(placements)} object(s). Wrapper at: {wrapper_xml}")
        if args.wrapper_out or args.keep:
            pass  # keep files
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


    finally:
        # Keep the temp folder for debugging by commenting this next line
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
