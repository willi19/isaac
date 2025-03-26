import os
from dex_robot.utils.file_io import load_camparam, load_robot_traj, load_c2r
import numpy as np
from dex_robot.simulate.simulator import simulator
import tqdm
import subprocess
import argparse

root_path = "/home/temp_id/shared_data/processed"
obj_list = os.listdir(root_path)

save_video = True
save_state = False
view_physics = False
view_replay = True
headless = True

simulator = simulator(
    None,
    view_physics,
    view_replay,
    headless,
    save_video,
    save_state,
    fixed=True,
    add_plane=False
)

if __name__ == "__main__":
    first = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, nargs="+", default=None, help="Object name(s) to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    if args.name is None:
        args.name = os.listdir(root_path)

    for obj_name in args.name:
        try:
            index_list = os.listdir(f"{root_path}/{obj_name}")
            for index in index_list:
                demo_path = f"{root_path}/{obj_name}/{index}"
                temp_path = f"video/projection/{obj_name}_temp/{index}"
                
                intrinsic, extrinsic = load_camparam(demo_path)
                cam_dict = {}
                try:
                    robot_traj = load_robot_traj(demo_path)
                except:
                    print(f"Failed to load robot trajectory: {demo_path}")
                    continue

                vid_path = temp_path.replace("_temp", "")
                if os.path.exists(vid_path) and not args.overwrite and save_video:
                    print(f"Skipping existing file: {vid_path}")
                    continue

                os.makedirs(temp_path, exist_ok=True)
                C2R = load_c2r(demo_path)
                
                cnt = 3
                for serial_num, param in intrinsic.items():
                    int_mat = np.array(param['Intrinsics']).reshape(3,3)
                    ext_mat = np.array(extrinsic[serial_num])
                    ext_mat = np.concatenate([ext_mat, np.array([[0,0,0,1]])], axis=0)
                    ext_mat = np.linalg.inv(C2R) @ np.linalg.inv(ext_mat)
                    ext_mat = ext_mat[:3]
                    cam_dict[serial_num] = (int_mat, ext_mat)
                    cnt -= 1
                    # if cnt == 0:
                    #     s_tmp = serial_num
                if first:
                    simulator.load_camera(cam_dict)
                    simulator.visualize_camera(cam_dict)

                    if not headless:
                        simulator.set_viewer(cam_dict["22684755"])
                simulator.set_savepath(temp_path, f"result/{obj_name}/{index}.mp4")
                T = robot_traj.shape[0]
                for step in tqdm.tqdm(range(T)):
                    state = robot_traj[step]
                    simulator.step(state, state, None)
                simulator.save()
                for vid_name in os.listdir(temp_path):
                    os.makedirs(vid_path, exist_ok=True)
                    temp_video_path = os.path.join(temp_path, vid_name)
                    output_video_path = os.path.join(vid_path, vid_name)
                    print(vid_name)
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",  # 기존 파일 덮어쓰기
                        "-i", temp_video_path,  # 입력 파일
                        "-c:v", "libx264",  # 비디오 코덱: H.264
                        "-preset", "slow",  # 압축률과 속도 조절 (slow = 고품질)
                        "-crf", "23",  # 품질 설정 (낮을수록 고품질, 18~23 추천)
                        "-pix_fmt", "yuv420p",  # 픽셀 포맷 (H.264 표준 호환)
                        output_video_path
                    ]

                    # FFmpeg 실행
                    subprocess.run(ffmpeg_cmd, check=True)
                    print(f"✅ H.264 encoded video saved: {output_video_path}")
                    os.remove(temp_video_path)  # 변환 후 임시 파일 삭제
                    
                os.removedirs(temp_path)
                first = False
        except Exception as e:
            print(f"Error processing {demo_path}: {e}")
            continue
        
        if os.path.exists(f"video/projection/{obj_name}_temp"):
            os.removedirs(f"video/projection/{obj_name}_temp")