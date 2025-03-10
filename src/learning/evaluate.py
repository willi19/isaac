

def sample(data_root, exp_dir, ckpt_step=1000):
    ckpt_filename = f"{ckpt_step}.pth"
    sample_dir = os.path.join(exp_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    ycb_model_dir = os.path.join(os.path.join(data_root, "processed"), "models")

    # selected_ycb_ids = list(YCB_CLASSES.keys())
    selected_ycb_ids = [1, 3, 4, 5, 9, 10, 11, 12, 14, 15, 19, 21]
    obj_pts_dataset = []
    for ycb_id in selected_ycb_ids:
        ycb_name = YCB_CLASSES[ycb_id]
        if ycb_name != "100_ball":
            mesh = trimesh.load(os.path.join(ycb_model_dir, ycb_name, "points.xyz"))
            pts = np.asarray(mesh.vertices)
        else:
            pts = np.asarray(trimesh.creation.icosphere(radius=0.05, subdivisions=4).vertices)
        sampled_pts = sample_data(pts, sample_num=2000)
        obj_pts_dataset.append(torch.from_numpy(sampled_pts).float())

    model = SeqAllegroQpos2AllegroCVAET()
    model = model.cuda()
    model = model.eval()

    ckpt_dict = torch.load(os.path.join(exp_dir, ckpt_filename))
    model.load_state_dict(ckpt_dict["model"])

    results, codes = [], []
    for obj_pts in tqdm.tqdm(obj_pts_dataset):
        obj_pts = torch.transpose(obj_pts.unsqueeze(0), 1, 2).cuda()
        batch_size = obj_pts.shape[0]

        query_t = repeat(torch.linspace(1, 0, 40), "n -> b n 1", b=batch_size).cuda()
        _result, _code = [], []

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        for _ in range(20000):
            with torch.no_grad():
                pred, z = model.sample(obj_pts, query_t, return_code=True)

            if not torch.isfinite(pred).all():
                print("pred contains INF or NaN!")

            pred_rot_6d = pred[..., 3 : 3 + 6]
            pred_rot = matrix_to_axis_angle(rotation_6d_to_matrix(pred_rot_6d))
            pred = torch.cat([pred[..., :3], pred_rot, pred[..., 3 + 6 :]], dim=-1)

            _result.append(pred[0].detach().cpu().numpy())
            _code.append(z[0].detach().cpu().numpy())
        results.append(_result)
        codes.append(_code)

    results, codes = np.asarray(results), np.asarray(codes)
    np.save(os.path.join(sample_dir, "result.npy"), results)
    np.save(os.path.join(sample_dir, "code.npy"), codes)
    print(f"Save result to {os.path.join(sample_dir, 'result.npy')}")
    print(f"Save code to {os.path.join(sample_dir, 'code.npy')}")

    print("Filtering the results...")
    z = results[..., 2]
    mask = np.any(z > 0.1, axis=-1)

    aa = results[..., -1, 3:6]
    axis = split_axis_angle(aa)[0]
    angle_mask = np.dot(axis, np.array([0, 0, 1])) < 0.85
    mask = np.logical_and(mask, angle_mask)

    filter_result = {}
    filter_code = {}
    for i in range(len(results)):
        filter_result[YCB_CLASSES[selected_ycb_ids[i]]] = results[i, mask[i]]
        filter_code[YCB_CLASSES[selected_ycb_ids[i]]] = codes[i, mask[i]]
    os.makedirs(os.path.join(sample_dir, "result_filter"), exist_ok=True)
    os.makedirs(os.path.join(sample_dir, "code_filter"), exist_ok=True)
    for k, qpos in filter_result.items():
        qpos = qpos.copy()
        filter_result_path = os.path.join(sample_dir, "result_filter", f"{k}.npy")
        filter_code_path = os.path.join(sample_dir, "code_filter", f"{k}.npy")
        np.save(filter_result_path, qpos)
        np.save(filter_code_path, filter_code[k])

    print("Process result for the sapien simulator...")
    filter_result_sapien = {}
    for k, qpos in filter_result.items():
        obj_tl = np.array([0, 0, YCB_SIZE[k][2] / 2])
        init_pos = np.array([0.4, 0, -0.2])
        qpos = qpos.copy()

        qpos[..., :3] += obj_tl
        qpos[..., :3] += init_pos

        rot_mat = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        qpos[..., :3] = rotate(qpos[..., :3], rot_mat)
        qpos[..., 3:6] = matrix_to_euler_angles(rot_mat @ axis_angle_to_matrix(qpos[..., 3:6]), convention="rxyz")
        filter_result_sapien[k] = qpos
    os.makedirs(os.path.join(sample_dir, "result_filter_sapien"), exist_ok=True)
    for k in filter_result_sapien:
        filter_result_path = os.path.join(sample_dir, "result_filter_sapien", f"{k}.npy")
        np.save(filter_result_path, filter_result_sapien[k])

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--mano_side", type=str, choices=["left", "right"], default="left")

    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train")
    parser.add_argument("--ts", type=str, default=None)
    parser.add_argument("--ckpt_step", type=int, default=1000)
    args = parser.parse_args()

    assert args.ts is not None, "Please specify the timestamp for sampling"
    assert os.path.exists(os.path.join(args.output_dir, args.ts)), "The timestamp does not exist"
    sample(args.data_root, exp_dir=os.path.join(args.output_dir, args.ts), ckpt_step=args.ckpt_step)