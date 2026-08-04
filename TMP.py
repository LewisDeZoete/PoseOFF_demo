#!/usr/bin/env python3
import numpy as np
import cv2
from scipy.ndimage import sobel, convolve1d, gaussian_filter
from einops import rearrange
import av


def temporal_gradient_5(video) -> np.ndarray:
    """Compute temporal gradients using 5-point stencil central difference.

    Uses the formula: f'(t) ≈ [-f(t+2) + 8f(t+1) - 8f(t-1) + f(t-2)] / (12)

    For boundary frames (first 2 and last 2), falls back to lower-order
    central differences since the full stencil isn't available.

    Args:
        video (np.ndarray): Array of shape (T, C, H, W).

    Returns:
        grad (np.ndarray): Array of shape (T, C, H, W), same dtype family as input.
    """
    T = video.shape[0]
    if T < 2:
        raise ValueError("Need at least 2 frames to compute a temporal gradient.")

    video = video.astype(np.float32, copy=False) if video.dtype != np.float64 else video
    It = np.empty_like(video, dtype=video.dtype)

    if T >= 5:
        # Interior points: 5-point stencil, vectorized over the whole interior at once
        It[2:-2] = (-video[4:] + 8*video[3:-1] - 8*video[1:-3] + video[0:-4]) / (12)

    # Boundary handling
    if T >= 5:
        # second point and second-to-last: 3-point central difference
        It[1] = (video[2] - video[0]) / (2)
        It[-2] = (video[-1] - video[-3]) / (2)
    elif T >= 3:
        # T == 3 or 4: use 3-point central diff for all interior points
        It[1:-1] = (video[2:] - video[:-2]) / (2)

    # First and last frame: forward/backward difference (one-sided)
    It[0] = (video[1] - video[0])
    It[-1] = (video[-1] - video[-2])

    return It

def spatial_gradient(grey):
    """Calculate the spatial gradient for an input array representing a (greyscale) video"""

    Ix = sobel(grey, axis=2)
    Iy = sobel(grey, axis=2)

    return Ix, Iy

def get_normal_flow(
        video,
        sigma: float = 1.0,
        mag_threshold: float = 100.0
):
    """Compute normal flow by calculating spatial and temporal gradients.

    Args:
        video (np.ndarray/torch.tensor): input video of shape (T, C, H, W)
        sigma (float): spatial gaussian smoothing applied before differencing. Default is 1.0.
        mag_threshold (floats): upper magnitude threshold. Default is 100.0.

    Returns:
        norm_flows (np.array): Normal flows of shape (n_frames-1, 2, H, W)
    """
    # Collapse channels (T, H, W)
    grey = video.mean(axis=1)

    # Optionally apply Gaussian smoothing
    if sigma > 0:
        grey = gaussian_filter(grey, sigma=(0, sigma, sigma))

    # --- Temporal gradient (T, H, W) ---
    It = temporal_gradient_5(grey)

    # --- Spatial gradients (T, H, W) ---
    Ix, Iy = spatial_gradient(grey)

    # --- Normal flow calculations ---
    # Create a mask for removing gradients flows that are too small
    denom = Ix**2 + Iy**2
    denom_safe = np.where(denom == 0, 1.0, denom)
    scale = np.where(denom ==0, 0.0, -It / denom_safe)

    norm_flow = np.stack([scale * Ix, scale * Iy], axis=-1) # (T-1, H, W, 2)

    # Mask pixels whee flow magnitude exceeds the threshold
    flow_mag = np.sqrt((norm_flow**2).sum(axis=-1, keepdims=True))
    norm_flow = np.where(flow_mag > mag_threshold, 0.0, norm_flow)

    return norm_flow

def load_video_numpy(path: str, max_frames: int | None = None) -> np.ndarray:
    """Load a video into a numpy array of shape (T, H, W, C) uint8."""
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if max_frames is not None and i >= max_frames:
                break
            # frame.to_ndarray give (H, W, C) in RGB by default!
            frames.append(frame.to_ndarray(format="rgb24"))

    return np.stack(frames, axis=0)


if __name__ == "__main__":
    from utils import draw_flow_hsv, draw_flow_arrows

    video = load_video_numpy(path="./input/walk.mp4")

    video = rearrange(video, "T H W C -> T C H W")

    # --- Normal flow calculations here ---
    normal_flow = get_normal_flow(video)

    print("Computed normal flow!")

    # Annoyingly, convert the video shape back...
    video = rearrange(video, "T C H W -> T H W C")

    flow_frames = [draw_flow_arrows(frame, normal_flow[frame_no]) for frame_no, frame in enumerate(video)]

    print(flow_frames[0].shape)

    cv2.imshow("HELLO???", video[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # for frame_no, frame in enumerate(video):

    #     normal_flow_frame = normal_flow[frame_no].astype(np.float32)
    #     print(f"Normal flow frame shape: { normal_flow_frame.shape }")

    #     frame = draw_flow_arrows(frame, normal_flow_frame).astype(np.uint8)
    #     print(f"Drew to frame: {frame.shape} / {frame.dtype}")


    #     cv2.imshow("Frame", frame)
    #     keypress = cv2.waitKey(1)
    #     if keypress == ord('q'):
    #         print("exiting video playback")
    #         break

    # cv2.destroyAllWindows()
