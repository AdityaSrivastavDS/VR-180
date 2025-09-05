import os, math, tempfile, time
import cv2
import numpy as np
from tqdm import tqdm
import torch

# ---------------- Load MiDaS ----------------
def load_midas_model(use_fast=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = "MiDaS_small" if use_fast else "DPT_Large"
    try:
        model = torch.hub.load("intel-isl/MiDaS", model_type)
        model.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform
        return model, transform, device
    except Exception as e:
        raise RuntimeError("Failed to load MiDaS model. Error: " + str(e))

# ---------------- Depth Estimation ----------------
def estimate_depth_for_frame(frame_bgr, model, transform, device, prev_depth=None, alpha=0.6):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img).to(device)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)

    # Temporal smoothing to reduce shaking
    if prev_depth is not None:
        depth_norm = alpha * prev_depth + (1 - alpha) * depth_norm

    # Small blur to smooth local noise
    depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), 0)
    return depth_norm

# ---------------- Stereo Pair Synthesis ----------------
def synthesize_stereo_pair(rgb, depth_norm, max_shift=40):
    h, w = depth_norm.shape
    disp = (depth_norm * max_shift).astype(np.float32)
    disp = cv2.GaussianBlur(disp, (5, 5), 0)  # smooth disparity map

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (xx - disp).astype(np.float32)
    map_y = yy.astype(np.float32)

    right = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)  # better than black edges
    left = rgb.copy()
    return left, right

# ---------------- Main Converter ----------------
def convert_video_to_vr180(input_path, output_path, max_width=960, frame_step=1,
                           max_shift=40, use_fast_midas=True, progress_callback=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, transform, device = load_midas_model(use_fast=use_fast_midas, device=device)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames_to_process = list(range(0, total_frames, frame_step))
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = None
    prev_depth = None
    processed = 0

    for i, fidx in enumerate(frames_to_process):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        scale = max_width / float(w) if w > max_width else 1.0
        if scale != 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            h, w = frame.shape[:2]

        depth = estimate_depth_for_frame(frame, model, transform, device, prev_depth)
        prev_depth = depth.copy()

        left, right = synthesize_stereo_pair(frame, depth, max_shift)
        side_by_side = np.concatenate([left, right], axis=1)

        if writer is None:
            writer = cv2.VideoWriter(tmp_out, fourcc, fps / frame_step,
                                     (side_by_side.shape[1], side_by_side.shape[0]))
        writer.write(side_by_side)

        processed += 1
        if progress_callback:
            progress_callback(processed/len(frames_to_process), f'Frame {i+1}/{len(frames_to_process)}')

    if writer:
        writer.release()
    cap.release()
    os.replace(tmp_out, output_path)
    return output_path

def sample_preview_frames(input_path, max_width=960):
    try:
        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        # Sample two frames near start and middle
        sample_idxs = [max(0, min(total-1, total//10)),
                       max(0, min(total-1, total//2))]

        for idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            scale = max_width / float(w) if w > max_width else 1.0
            if scale != 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        if len(frames) >= 2:
            tmp = tempfile.mkdtemp()
            p1 = os.path.join(tmp, 'preview_orig.png')
            p2 = os.path.join(tmp, 'preview_gray.png')
            cv2.imwrite(p1, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            gray = cv2.cvtColor(frames[1], cv2.COLOR_RGB2GRAY)
            gray_col = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(p2, cv2.cvtColor(gray_col, cv2.COLOR_RGB2BGR))
            return [p1, p2]
        return []
    except Exception:
        return []
