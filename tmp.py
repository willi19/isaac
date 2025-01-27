from dex_robot.retargeting.retargeting_config import RetargetingConfig

config_path = "teleop/xarm6_allegro_hand_right_position.yml"

config = RetargetingConfig.load_from_file(config_path)
