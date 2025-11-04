from robosuite.utils.mjcf_utils import new_site, new_joint, new_actuator, new_tendon, new_geom
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange
import numpy as np


@register_problem
class CDPRDomain(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        # Configure the workspace for CDPR
        self.workspace_name = "cdpr_workspace"
        self.visualization_sites_list = []
        
        # CDPR specific parameters
        self.frame_points = np.array([
            [-0.535, -0.755, 1.309],
            [0.755, -0.525, 1.309],
            [0.535, 0.755, 1.309],
            [-0.755, 0.525, 1.309]
        ])
        
        # Update kwargs for CDPR setup
        kwargs.update({"robots": []})  # No robots, we'll use CDPR actuators
        kwargs.update({"workspace_offset": (0, 0, 0)})
        kwargs.update({"arena_type": "cdpr"})
        
        # Specify custom scene XML
        kwargs.update({"scene_xml": "scenes/cdpr_scene.xml"})
        kwargs.update({"scene_properties": {
            "floor_style": "gray-ceramic",
            "wall_style": "light-gray",
        }})

        super().__init__(bddl_file_name, *args, **kwargs)
        
        # CDPR control parameters
        self.Kp = 100
        self.Kd = 130
        self.threshold = 0.03
        self.target_pos = np.array([0, 0, 1.309])
        self.prev_lengths = np.zeros(4)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Load CDPR fixtures (frame, rotors, sliders)"""
        # CDPR frame structure will be loaded from the scene XML
        pass

    def _load_objects_in_arena(self, mujoco_arena):
        """Load the end-effector and target object"""
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        """Load sites for visualization and target regions"""
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        
        for object_region_name in list(region_dict.keys()):
            # Handle target regions
            ranges = region_dict[object_region_name]["ranges"][0]
            zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
            zone_centroid_xy = (
                (ranges[2] + ranges[0]) / 2,
                (ranges[3] + ranges[1]) / 2,
            )
            target_zone = TargetZone(
                name=object_region_name,
                rgba=region_dict[object_region_name]["rgba"],
                zone_size=zone_size,
                z_offset=0.01,
                zone_centroid_xy=zone_centroid_xy,
            )
            object_sites_dict[object_region_name] = target_zone
            
            mujoco_arena.table_body.append(
                new_site(
                    name=target_zone.name,
                    pos=target_zone.pos,
                    quat=target_zone.quat,
                    rgba=target_zone.rgba,
                    size=target_zone.size,
                    type="box",
                )
            )
        
        self.object_sites_dict = object_sites_dict

    def inverse_kinematics(self, pos):
        """Calculate cable lengths for given end-effector position"""
        lengths = []
        for i in range(4):
            vec = pos - self.frame_points[i]
            lengths.append(np.linalg.norm(vec))
        return np.array(lengths)

    def _setup_references(self):
        """Set up references for CDPR components"""
        super()._setup_references()
        
        # Store references to CDPR components
        self.slider_joints = [
            self.sim.model.joint_name2id(f"slider_{i+1}") for i in range(4)
        ]
        self.ee_body = self.sim.model.body_name2id("box")
        
        # Initialize previous lengths
        ee_pos = self.sim.data.body_xpos[self.ee_body]
        self.prev_lengths = self.inverse_kinematics(ee_pos)

    def _setup_controllers(self):
        """Setup CDPR controller"""
        # CDPR control is handled in _pre_action
        pass

    def _pre_action(self, action, policy_step=False):
        """CDPR control logic"""
        # Get current end-effector position
        ee_pos = self.sim.data.body_xpos[self.ee_body]
        
        # Get current and target cable lengths
        cur_lengths = self.inverse_kinematics(ee_pos)
        target_lengths = self.inverse_kinematics(self.target_pos)
        
        # PD Control
        length_errors = target_lengths - cur_lengths
        cable_velocities = (cur_lengths - self.prev_lengths) / self.dt
        
        control_signals = self.Kp * length_errors - self.Kd * cable_velocities
        
        # Apply control to sliders
        for i in range(4):
            self.sim.data.ctrl[i] = -control_signals[i]
        
        self.prev_lengths = cur_lengths.copy()
        
        return super()._pre_action(action, policy_step)

    def set_target_position(self, target_pos):
        """Set new target position for CDPR"""
        if all(-1.309 <= coord <= 1.309 for coord in target_pos):
            self.target_pos = np.array(target_pos)
            # Reset previous lengths for derivative term
            ee_pos = self.sim.data.body_xpos[self.ee_body]
            self.prev_lengths = self.inverse_kinematics(ee_pos)
            return True
        return False

    def _check_success(self):
        """Check if end-effector reached target"""
        ee_pos = self.sim.data.body_xpos[self.ee_body]
        cur_lengths = self.inverse_kinematics(ee_pos)
        target_lengths = self.inverse_kinematics(self.target_pos)
        
        length_errors = np.abs(target_lengths - cur_lengths)
        return np.all(length_errors < self.threshold)

    def _setup_camera(self, mujoco_arena):
        """Configure cameras for CDPR"""
        # End-effector camera
        mujoco_arena.set_camera(
            camera_name="ee_camera",
            pos=[0, 0, -0.1],
            quat=[1, 0, 0, 0],
        )
        
        # Overview camera
        mujoco_arena.set_camera(
            camera_name="overview",
            pos=[2.0, 2.0, 2.0],
            quat=[0.707, 0.0, 0.0, 0.707],
        )