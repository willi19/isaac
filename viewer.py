import os
import numpy as np
from isaacgym import gymapi, gymtorch

# 초기화
gym = gymapi.acquire_gym()

# 시뮬레이션 설정
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 0.016
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 자산 경로
asset_root = os.path.join(os.getcwd(), "assets")
robot_asset_file = "xarm/xarm.urdf"
object_asset_file = "objects/bottle/bottle.obj"

# 로봇 및 객체 자산 로드
robot_asset = gym.load_asset(sim, asset_root, robot_asset_file, gymapi.AssetOptions())
object_asset = gym.load_asset(sim, asset_root, object_asset_file, gymapi.AssetOptions())

# 환경 생성
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

# 로봇 추가
robot_pose = gymapi.Transform()
robot_pose.p = gymapi.Vec3(0, 0, 0.5)
robot_handle = gym.create_actor(env, robot_asset, robot_pose, "robot", 0, 1)

# 객체 추가
object_pose = gymapi.Transform()
object_pose.p = gymapi.Vec3(0.5, 0.0, 0.1)
object_handle = gym.create_actor(env, object_asset, object_pose, "object", 0, 1)

# 관절 설정
joint_p_gain = np.array([1000.0] * 7, dtype=np.float32)
joint_d_gain = np.array([100.0] * 7, dtype=np.float32)

# 시뮬레이션 루프
for step in range(1000):
    # 관찰값 수집
    obs_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))

    # 행동 설정 (예제: 랜덤 행동)
    action = np.random.uniform(-1, 1, 7).astype(np.float32)
    gym.set_actor_dof_position_targets(sim, robot_handle, action)

    # 시뮬레이션 실행
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 결과 확인
    print(f"Step {step}: Observation - {obs_tensor[:3]}")

# 시뮬레이션 종료
gym.destroy_sim(sim)
