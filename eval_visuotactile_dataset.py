"""
Evaluate ProHMR on a prepared MyFusion visuotactile dataset source.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROHMR_ROOT = Path(__file__).resolve().parent / "ProHMR"
if str(PROHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(PROHMR_ROOT))

from ProHMR.prohmr.configs import get_config, prohmr_config
from ProHMR.prohmr.datasets.utils import get_example
from ProHMR.prohmr.utils import recursive_to
from models import ProHMR
from utils import Evaluator, dataset_config


MYFUSION_CACHE_ROOT = Path("/tmp/pear_myfusion_cache")

VisuotactileDatasetSource = None
prepared_files = None


def setup_myfusion(myfusion_path):
    global VisuotactileDatasetSource, prepared_files
    path = Path(myfusion_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MyFusion path does not exist: {path}")
    import_root = path.parent if path.name == "MyFusion" else path
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    from MyFusion.data_source.source_visuotactile_dataset import (
        VisuotactileDatasetSource as _VisuotactileDatasetSource,
        prepared_files as _prepared_files,
    )

    VisuotactileDatasetSource = _VisuotactileDatasetSource
    prepared_files = _prepared_files
    return import_root


def source_camera_count(source):
    data = source.data
    if "camera_names" in data.files:
        return int(np.asarray(data["camera_names"]).shape[0])
    if "camera_intrinsics_k" in data.files:
        intrinsics = np.asarray(data["camera_intrinsics_k"])
        if intrinsics.ndim >= 3:
            return int(intrinsics.shape[1])
        if intrinsics.ndim == 2:
            return int(intrinsics.shape[0])
    if getattr(source, "rgb", None) is not None:
        rgb = np.asarray(source.rgb)
        if rgb.ndim >= 5:
            return int(rgb.shape[1])
    return 1


def load_source_rgb_array(source, frame_idx):
    if getattr(source, "rgb_files", None) is not None:
        path = Path(source.rgb_files[frame_idx])
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is None:
                raise IOError(f"Fail to read {path}")
            return bgr[:, :, ::-1].astype(np.float32)
        loaded = np.load(path, mmap_mode="r")
        return np.asarray(loaded[loaded.files[0]] if hasattr(loaded, "files") else loaded)
    if getattr(source, "rgb", None) is None:
        raise FileNotFoundError(f"MyFusion source {source.path} has no RGB data")
    return np.asarray(source.rgb[frame_idx])


def source_camera_rgb(source, frame_idx, cam_idx, n_cams):
    rgb = load_source_rgb_array(source, frame_idx)
    if rgb.ndim == 4:
        return rgb[cam_idx].astype(np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame from MyFusion source, got shape {rgb.shape}")
    width = rgb.shape[1] // n_cams
    if width <= 0:
        raise ValueError(f"Cannot split RGB frame into {n_cams} horizontal camera views")
    return rgb[:, cam_idx * width : (cam_idx + 1) * width].astype(np.float32)


def apply_occlusion_texture(image, source, frame_idx, cam_idx):
    try:
        occl_path = source._occlusion_mask_path(frame_idx, cam_idx)
    except AttributeError:
        root = Path(source.path) if Path(source.path).is_dir() else Path(source.path).parent
        occl_path = root / "occlusion_patches" / f"cam{int(cam_idx) + 1}" / f"frame_{int(frame_idx):08d}_texture.png"
    if not Path(occl_path).exists():
        return image
    occl_bgr = cv2.imread(str(occl_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if occl_bgr is None:
        return image
    occl_rgb = occl_bgr[:, :, ::-1].astype(np.uint8)
    if occl_rgb.shape[:2] != image.shape[:2]:
        occl_rgb = cv2.resize(occl_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.any(occl_rgb != 0, axis=-1)
    if not mask.any():
        return image
    out = image.copy()
    out[mask] = occl_rgb[mask].astype(out.dtype)
    return out


def array_at(data, names, idx, default=None):
    for name in names:
        if name not in data.files:
            continue
        value = np.asarray(data[name])
        if value.ndim >= 2 and value.shape[0] > idx:
            return value[idx].astype(np.float32, copy=False)
        return value.astype(np.float32, copy=False)
    return default


def smpl_pose_at(data, idx, num_pose):
    pose = array_at(data, ("body_pose", "smpl_pose", "pose", "poses", "full_pose"), idx)
    if pose is None and "global_orientation" in data.files and "thetas" in data.files:
        pose = np.concatenate(
            [
                array_at(data, ("global_orientation",), idx).reshape(-1),
                array_at(data, ("thetas",), idx).reshape(-1),
            ]
        )
    out = np.zeros(num_pose, dtype=np.float32)
    has_pose = 0.0
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        out[: min(num_pose, pose.size)] = pose[:num_pose]
        has_pose = 1.0
    return out, np.float32(has_pose)


def smpl_betas_at(data, idx):
    betas = array_at(data, ("betas", "smpl_betas", "shape"), idx)
    out = np.zeros(10, dtype=np.float32)
    has_betas = 0.0
    if betas is not None:
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        out[: min(10, betas.size)] = betas[:10]
        has_betas = 1.0
    return out, np.float32(has_betas)


def camera_intrinsics(source, frame_idx, cam_idx):
    k_all = np.asarray(source.data["camera_intrinsics_k"][frame_idx], dtype=np.float32)
    k = k_all[cam_idx] if k_all.ndim >= 3 else k_all
    return k.reshape(3, 3)


def project_base_points_to_camera(points, extrinsics, intrinsics):
    points = np.asarray(points, dtype=np.float32)
    ext = np.asarray(extrinsics, dtype=np.float32)
    cam = (points - ext[:3, 3]) @ ext[:3, :3]
    valid = np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1e-6)
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    uv[:, 0] = intrinsics[0, 0] * cam[:, 0] / np.maximum(cam[:, 2], 1e-6) + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * cam[:, 1] / np.maximum(cam[:, 2], 1e-6) + intrinsics[1, 2]
    return uv, valid


def bbox_from_points(uv, valid, width, height, margin=1.35):
    valid &= np.isfinite(uv).all(axis=1)
    valid &= uv[:, 0] >= 0
    valid &= uv[:, 0] < width
    valid &= uv[:, 1] >= 0
    valid &= uv[:, 1] < height
    if not np.any(valid):
        size = float(max(width, height))
        return np.array([width * 0.5, height * 0.5, size], dtype=np.float32)
    pts = uv[valid]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    size = max(float((hi - lo).max() * margin), 32.0)
    return np.array([center[0], center[1], size], dtype=np.float32)


class MyFusionProHMRDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, input_path, camera_index=0, activate_occlusion=False):
        self.cfg = cfg
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
        self.activate_occlusion = bool(activate_occlusion)
        self.frames = []
        self.sources = []
        self.num_pose = 3 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.keypoint_count = 44

        for sequence_path in prepared_files(str(input_path)):
            source = VisuotactileDatasetSource(
                path=Path(sequence_path),
                cache_root=MYFUSION_CACHE_ROOT,
                load_rgb=True,
                rebuild_cache=False,
                max_points=1,
                use_tactile=False,
                occlusion=self.activate_occlusion,
            )
            self.sources.append(source)
            n_cams = source_camera_count(source)
            selected_cams = list(range(n_cams)) if int(camera_index) < 0 else [int(camera_index)]
            if any(cam < 0 or cam >= n_cams for cam in selected_cams):
                raise ValueError(f"camera_index must be in [0,{n_cams - 1}] or -1, got {camera_index}")
            for frame_idx in range(len(source)):
                for cam_idx in selected_cams:
                    self.frames.append((source, frame_idx, cam_idx, n_cams))

        if not self.frames:
            raise ValueError(f"No frames found in MyFusion input path: {input_path}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        source, frame_idx, cam_idx, n_cams = self.frames[idx]
        image_rgb = source_camera_rgb(source, frame_idx, cam_idx, n_cams)
        if self.activate_occlusion and cam_idx == 0:
            image_rgb = apply_occlusion_texture(image_rgb, source, frame_idx, cam_idx)
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1].astype(np.uint8))
        img_h, img_w = image_bgr.shape[:2]

        target = np.asarray(source.target_joints[frame_idx], dtype=np.float32)
        if target.shape[-1] == 3:
            target = np.concatenate([target, np.ones((target.shape[0], 1), dtype=np.float32)], axis=1)
        keypoints_3d = np.zeros((self.keypoint_count, 4), dtype=np.float32)
        target_indices = np.asarray(source.target_indices, dtype=np.int64).reshape(-1)
        if target_indices.size == target.shape[0] and target_indices.max(initial=-1) < self.keypoint_count:
            keypoints_3d[target_indices] = target
        else:
            count = min(self.keypoint_count, target.shape[0])
            keypoints_3d[:count] = target[:count]

        intrinsics = camera_intrinsics(source, frame_idx, cam_idx)
        extrinsics = source._retrieve_extrinsics(cam_idx)
        uv, visible = project_base_points_to_camera(keypoints_3d[:, :3], extrinsics, intrinsics)
        center_x, center_y, bbox_size = bbox_from_points(uv, visible & (keypoints_3d[:, 3] > 0), img_w, img_h)
        keypoints_2d = np.zeros((self.keypoint_count, 3), dtype=np.float32)
        keypoints_2d[:, :2] = uv
        keypoints_2d[:, 2] = (visible & (keypoints_3d[:, 3] > 0)).astype(np.float32)

        body_pose, has_body_pose = smpl_pose_at(source.data, frame_idx, self.num_pose)
        betas, has_betas = smpl_betas_at(source.data, frame_idx)
        smpl_params = {
            "global_orient": body_pose[:3],
            "body_pose": body_pose[3:],
            "betas": betas,
        }
        has_smpl_params = {
            "global_orient": has_body_pose,
            "body_pose": has_body_pose,
            "betas": has_betas,
        }
        smpl_params_is_axis_angle = {
            "global_orient": True,
            "body_pose": True,
            "betas": False,
        }

        tmp_img = f"/tmp/prohmr_myfusion_{os.getpid()}_{idx}.png"
        cv2.imwrite(tmp_img, image_bgr)
        img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size = get_example(
            tmp_img,
            float(center_x),
            float(center_y),
            float(bbox_size),
            float(bbox_size),
            keypoints_2d,
            keypoints_3d,
            smpl_params,
            has_smpl_params,
            list(range(self.keypoint_count)),
            self.img_size,
            self.img_size,
            self.mean,
            self.std,
            False,
            self.cfg.DATASETS.CONFIG,
        )

        return {
            "img": img_patch.astype(np.float32),
            "keypoints_2d": keypoints_2d.astype(np.float32),
            "keypoints_3d": keypoints_3d.astype(np.float32),
            "orig_keypoints_2d": keypoints_2d.astype(np.float32),
            "box_center": np.array([center_x, center_y], dtype=np.float32),
            "box_size": np.float32(bbox_size),
            "img_size": 1.0 * img_size[::-1].copy(),
            "smpl_params": smpl_params,
            "has_smpl_params": has_smpl_params,
            "smpl_params_is_axis_angle": smpl_params_is_axis_angle,
            "imgname": f"{Path(source.path).stem}_frame{frame_idx:08d}_cam{cam_idx + 1}",
            "idx": idx,
        }


parser = argparse.ArgumentParser(description="Evaluate trained models on MyFusion visuotactile data")
parser.add_argument("--checkpoint", type=str, default="data/checkpoint.pt", help="Path to pretrained model checkpoint")
parser.add_argument("--model_cfg", type=str, default=None, help="Path to config file")
parser.add_argument("--dataset", type=str, default="3DPW-TEST-OC", help="Dataset config entry used for evaluation metadata")
parser.add_argument("--input_path", type=str, required=True, help="Prepared MyFusion folder, folder-of-folders, or .npz")
parser.add_argument("--myfusion_path", type=str, default="", required=True, help="Path containing the MyFusion module")
parser.add_argument("--camera_index", type=int, default=0, help="Camera to evaluate, 0 is cam1, -1 evaluates all cameras")
parser.add_argument("--activate_occlusion", action="store_true", help="Apply MyFusion occlusion textures to cam1 RGB inputs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
parser.add_argument("--num_measurements", type=int, default=3)
parser.add_argument("--num_samples", type=int, default=1024, help="Number of test samples to draw")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers used for data loading")
parser.add_argument("--log_freq", type=int, default=2500, help="How often to log results")
parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=False, help="Shuffle the dataset during evaluation")
parser.add_argument("--save_patch", action="store_true")

args = parser.parse_args()

setup_myfusion(args.myfusion_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if args.model_cfg is None:
    model_cfg = prohmr_config()
else:
    model_cfg = get_config(args.model_cfg)

dataset_cfg = dataset_config("configs/datasets.yaml")[args.dataset]

model_cfg.defrost()
model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
model_cfg.freeze()

model = ProHMR.load_from_checkpoint(args.checkpoint, strict=False, cfg=model_cfg).to(device)
model.eval()

dataset = MyFusionProHMRDataset(
    model_cfg,
    input_path=args.input_path,
    camera_index=args.camera_index,
    activate_occlusion=args.activate_occlusion,
)
dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

metrics = [
    "mode_mpjpe",
    "rmcs_mpjpe",
    "rmsf_mpjpe",
    "amcs_mpjpe",
    "amsf_mpjpe",
    "mode_re",
    "rmcs_re",
    "rmsf_re",
    "amcs_re",
    "amsf_re",
    "time_am",
    "time_sf",
]

evaluator = Evaluator(
    dataset_length=len(dataset),
    keypoint_list=dataset_cfg.KEYPOINT_LIST,
    pelvis_ind=model_cfg.EXTRA.PELVIS_IND,
    smpl=model.smpl,
    metrics=metrics,
    n_measure_points=args.num_measurements,
)


def get_elapsed_ms(start_time=None, start_event=None, end_event=None):
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    return (time.perf_counter() - start_time) * 1000


prohmr_inference_times = []
amcs_pipeline_times = []
amsf_pipeline_times = []

for i, batch in enumerate(tqdm(dataloader)):
    batch = recursive_to(batch, device)
    with torch.no_grad():
        if device.type == "cuda":
            prohmr_start = torch.cuda.Event(enable_timing=True)
            prohmr_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            prohmr_start.record()
        else:
            prohmr_start_time = time.perf_counter()
        out = model(batch)
        prohmr_time = get_elapsed_ms(
            start_time=prohmr_start_time if device.type != "cuda" else None,
            start_event=prohmr_start if device.type == "cuda" else None,
            end_event=prohmr_end if device.type == "cuda" else None,
        )
        prohmr_inference_times.append(prohmr_time)
    evaluator(out, batch, flow_net=model.flow, smpl=model.smpl)
    batch_size = out["pred_keypoints_3d"].shape[0]
    amcs_pipeline_times.append(prohmr_time + evaluator.time_am[evaluator.counter - batch_size])
    amsf_pipeline_times.append(prohmr_time + evaluator.time_sf[evaluator.counter - batch_size])
    avg_prohmr_time = sum(prohmr_inference_times) / len(prohmr_inference_times)
    avg_amcs_time = sum(amcs_pipeline_times) / len(amcs_pipeline_times)
    avg_amsf_time = sum(amsf_pipeline_times) / len(amsf_pipeline_times)
    tqdm.write(
        f"Iter {i + 1}: avg ProHMR inference: {avg_prohmr_time:.3f} ms | "
        f"avg AM-CS pipeline: {avg_amcs_time:.3f} ms | "
        f"avg AM-SF pipeline: {avg_amsf_time:.3f} ms"
    )
    if i % args.log_freq == args.log_freq - 1:
        evaluator.log()
evaluator.log()
