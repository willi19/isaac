import numpy as np
import matplotlib.pyplot as plt
from isaacgym import gymapi, gymutil
import os

# Initialize Isaac Gym
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

if sim is None:
    raise Exception("Failed to create simulation")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")
camera_position = np.array([2, 0, 1])

# Calculate direction vector (camera looks at the actor)
camera_direction = -camera_position
camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize

# Set viewer camera transform
gym.viewer_camera_look_at(
    viewer, None, gymapi.Vec3(*camera_position), gymapi.Vec3(0, 0, 0)
)

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# Load URDF and get the robot asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

asset_root = os.path.join(os.getcwd(), "rsc")
robot_asset_file = "xarm6/xarm6_allegro_wrist_mounted_rotate.urdf"

robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, asset_options)

# Add robot to the environment
robot_pose = gymapi.Transform()
robot_pose.p = gymapi.Vec3(0, 0, 0)
robot_actor = gym.create_actor(env, robot_asset, robot_pose, "robot", 0, 1)

# Get joint limits
joint_limits = []
for i in range(gym.get_asset_dof_count(robot_asset)):
    dof_properties = gym.get_asset_dof_properties(robot_asset)
    lower_limit = dof_properties["lower"][i]
    upper_limit = dof_properties["upper"][i]
    joint_limits.append((lower_limit, upper_limit))


# Sample random joint configurations
def sample_joint_configs(joint_limits, num_samples=1000):
    configs = []
    for _ in range(num_samples):
        config = [np.random.uniform(limit[0], limit[1]) for limit in joint_limits]
        configs.append(config)
    return configs


# Compute end-effector positions
def compute_ee_positions(env, robot_actor, joint_configs):
    ee_positions = []
    robot_dof_state = gym.get_actor_dof_states(env, robot_actor, gymapi.STATE_ALL)

    for config in joint_configs:
        # Apply joint configurations
        for i, joint_value in enumerate(config):
            robot_dof_state["pos"][i] = joint_value

        gym.set_actor_dof_states(env, robot_actor, robot_dof_state, gymapi.STATE_ALL)

        # Step the physics simulation to update state
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # Get the pose of the end-effector (assuming it's the last link)
        ee_pose = gym.get_actor_rigid_body_states(env, robot_actor, gymapi.STATE_POS)[
            -1
        ]
        ee_pos = ee_pose["pose"]["p"]
        ee_positions.append([ee_pos[0], ee_pos[1], ee_pos[2]])

    return np.array(ee_positions)


# Add markers to visualize end-effector positions
def add_markers(env, positions):
    marker_handles = []
    for i, pos in enumerate(positions):
        marker_pose = gymapi.Transform()
        marker_pose.p = gymapi.Vec3(pos[0], pos[1], pos[2])

        sphere_handle = gym.create_actor(
            env,
            gym.create_sphere(sim, 0.002, gymapi.AssetOptions()),
            marker_pose,
            "marker",
            i + 100,
            0,
        )
        marker_handles.append(sphere_handle)
    return marker_handles


# Main execution
num_samples = 10000  # Number of random samples (reduce if needed for performance)
joint_configs = sample_joint_configs(joint_limits, num_samples)
ee_positions = compute_ee_positions(env, robot_actor, joint_configs)

# Add markers for the action space
add_markers(env, ee_positions)

# Render the environment and robot
while not gym.query_viewer_has_closed(viewer):
    # gym.simulate(sim)
    # gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
