#!/usr/bin/env python3

from einops import rearrange
from ultralytics import YOLO
import torch
import numpy as np
import cv2
from collections import deque

# ---------------------------------------------------------
# EXTRACTION TOOLS
# ---------------------------------------------------------

def temporal_gradient_5point(frames: list[np.ndarray]) -> np.ndarray:
    '''Classical 5-point stencil central difference.
    Fourth-order accurate: O(Δt⁴).

    Kernel:  (-1/12, 8/12, 0, -8/12, 1/12)  applied to [I_{-2}, I_{-1}, I_0, I_1, I_2]
    Assumes the middle frame (index 2) is the current frame.
    Requires exactly 5 frames.
    '''
    assert len(frames) == 5, "5-point stencil requires exactly 5 frames"
    f = [f.astype(np.float32) for f in frames]
    # Numerator: -f[-2] + 8*f[-1] - 8*f[1] + f[2]  (normalised by 12)
    It = (-f[0] + 8*f[1] - 8*f[3] + f[4]) / 12.0
    return It


def temporal_gradient_sobel(frames: list[np.ndarray]) -> np.ndarray:
    '''Sobel filter for temporal gradient.

    Kernel: (-1/8, -2/8, 0, 2/8, 1/8) applied to [I_{-2}, I_{-1}, I_0, I_1, I_2]
    Assumes the middle frame (index 2) is the current frame.
    Requires exactly 5 frames.
    '''
    assert len(frames) == 5, "5-point stencil requires exactly 5 frames"
    f = [f.astype(np.float32) for f in frames]
    # Numerator: -f[-2] + 8*f[-1] - 8*f[1] + f[2]  (normalised by 12)
    It = (-f[0] - 2*f[1] + 2*f[3] + f[4]) / 8.0
    return It


def spatial_gradients(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = frame.astype(np.float32)
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    return Ix, Iy


def compute_normal_flow(
    Ix: np.ndarray,
    Iy: np.ndarray,
    It: np.ndarray,
    grad_thresholds: list[float] = [1.0, 500.0],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the normal flow (u_n, v_n) at each pixel.

    The normal flow vector at a pixel is:

        [u_n, v_n] = -I_t / (Ix² + Iy²)  *  [Ix, Iy]

    Pixels where |∇I| is below grad_threshold are masked out (flat regions
    where the aperture problem makes normal flow meaningless).

    Parameters
    ----------
    Ix, Iy  : spatial gradients (H, W)
    It      : temporal gradient (H, W)
    grad_threshold : minimum gradient magnitude to trust the estimate

    Returns
    -------
    u_n, v_n : normal flow components (H, W), NaN where masked
    mask     : boolean array, True where flow is valid
    """
    spatial_grads = np.sqrt(Ix**2 + Iy**2)
    mask = (spatial_grads > grad_thresholds[0]) & (spatial_grads < grad_thresholds[1])

    # Scalar normal speed: s = -I_t / |∇I|²
    s = np.where(mask, -It / (spatial_grads + 1e-6), 0.0)

    u_n = s * Ix    # x-component
    v_n = s * Iy    # y-component

    u_n[~mask] = np.nan
    v_n[~mask] = np.nan

    return u_n, v_n, mask

    
class NormalFlowEstimator:
    def __init__(
            self,
            buffer_size: int = 5,
            grad_thresholds: list[float] = [1.0, 100.0],
            temporal_estimator = temporal_gradient_5point,
            ksize: int = 9
    ):
        '''TODO: Docstring

        Args:
            buffer_size (int, optional): Size of frame buffer for normal flow. Defaults to 5.
            grad_threshold (list[float], optional): Minimum and maximum gradients, above/below which are ignored. Defaults to [1.0, 100.0].
            temporal_estimator (optional): Temporal gradient estimator, either 5pt central difference or sobel style. Defaults to temporal_gradient_5pt.
            ksize (int, optional): Gaussian blur kernal size - frames are blurred pre-norm flow estimation. Defaults to 9.

        Returns:
            norm_flow (np.ndarray): Array of calculated normal flow vectors of shape (H,W,2).
            frame (np.ndarray): Input RGB frame.
        '''
        self.buffer_size = buffer_size
        self.grad_thresholds = grad_thresholds
        self.temporal_estimator = temporal_estimator
        self.ksize = ksize
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._buffer_rgb: deque[np.ndarray] = deque(maxlen=buffer_size)

    def push(self, frame):
        self._buffer_rgb.append(frame)
        if frame.ndim == 3: # Ensure grey image...
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bit of Gaussian blur for the sake if it :)
        frame = cv2.GaussianBlur(frame, (self.ksize, self.ksize), 0)

        # put frame in frame buffer
        self._buffer.append(frame.astype(np.float32))
        if len(self._buffer) < self.buffer_size:
            return None, None # still warming up...
        frames = list(self._buffer)
        # It = temporal_gradient_5point(frames)
        It = self.temporal_estimator(frames)
        ref_frame = frames[2]
        Ix, Iy = spatial_gradients(ref_frame)

        u_n, v_n, mask = compute_normal_flow(Ix, Iy, It, self.grad_thresholds)

        norm_flow = np.stack([u_n, v_n], axis=-1)
        norm_flow = np.where(np.stack([mask, mask], -1), norm_flow, 0.0)

        return norm_flow, self._buffer_rgb[2]

    def offline(self, frames: list[np.ndarray]):
        """Offline normal flow calculation!

        Args:
            frames (list[np.ndarray]): Buffer of length self.buffer_size for which norm flow is calculated.

        Returns:
            norm_flow (np.ndarray): 2D array of normal flows calculated from frames.
            frame (np.ndarray): RGB frame at centre (index 2) of input frames.
        """
        assert len(frames) == self.buffer_size
        rgb_frame = frames[2]
        if frames[0].ndim == 3: # Ensure grey image...
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

        # Estimate temporal gradients
        It = self.temporal_estimator(frames)
        ref_frame = frames[2]
        Ix, Iy = spatial_gradients(ref_frame)

        u_n, v_n, mask = compute_normal_flow(Ix, Iy, It, self.grad_thresholds)

        norm_flow = np.stack([u_n, v_n], axis=-1)
        norm_flow = np.where(np.stack([mask, mask], -1), norm_flow, 0.0)

        return norm_flow, rgb_frame

    
def get_poses(frame, pose_model, threshold=0.2):
    results = pose_model(frame, verbose=False)
    result = results[0]
    # data output shape: ((x,y,conf.), keypoints, bodies)
    poses = torch.zeros(3, 17, 2)
    for m, person in enumerate(result.keypoints):
        if m >= 2:
            break
        try:
            assert person.xyn.shape[1] == 17
        except AssertionError:
            continue
        # YOLO pose output is in the interval [0-1]
        poses[0, :, m] = person.xyn[0, :, 0]
        poses[1, :, m] = person.xyn[0, :, 1]
        poses[2, :, m] = person.conf[0]

        # set x and y to zero if confidence is zero
        poses[0][poses[2] < threshold] = 0
        poses[1][poses[2] < threshold] = 0

    poses = rearrange(poses, 'C V M -> (M V) C')
    return poses


def poseoff_lk(frame1, frame2, poses, window_size=3, threshold=0.2, dilation=1):
    """Using the LK method of optical flow calculation...
    CV implementation: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    goodFeaturesToTrack returns list of length `max_corners`, of shape: [max_corners, 1, 2].
    For each corner, you can simply ravel to flatten the array and get (x,y) positions.
    NOTE: The raw poses (from denoised_skes_data) are of shape: (T, M, V, C)
        In the get_poseoff_samples.py loop, we reshape (poses = poses.transpose(3, 0, 2, 1)) -> (C, T, V, M)

    Args:
        frame1 (torch.Tensor): First frame (grey) of shape (H W)
        frame2 (torch.Tensor): Second frame (grey) of shape (H W)
        poses (torch.Tensor): Pose keypoint tensor of shape ((M V) C)
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): Threshold below which samples are discarded...
        dilation (int): The dilation factor for sampling points around keypoints. Default is 1.
        debug_frame (None/int): Optionally return the frame_number, the frame itself and
            the current state of the poseoff array. Default is None.

    Returns:
        poseoff_aray: Array containing only the flow windows of shape:
            (C*window_size**2, total_keypoints)
    """
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    half_k = window_size // 2
    pose_local = poses.detach().clone()

    # Get some shapes of input tensors
    height, width = frame1.shape
    total_keypoints, _ = poses.shape

    pose_local[:, 0] = pose_local[:, 0] * (width-1)
    pose_local[:, 1] = pose_local[:, 1] * (height-1)
    pose_local = rearrange(pose_local, '(M V) C -> C (M V)', M=2, V=17)

    # pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
    #             * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
    vis = pose_local[2, :].flatten() > threshold  # Visibility mask (frames, keypoints)

    # Exclude keypoints that are too close to the edge where the flow window is cut off
    valid_indices = (
        vis.reshape(total_keypoints) &
        (pose_local[0, :] >= half_k * dilation) &
        (pose_local[0, :] < width - half_k * dilation) &
        (pose_local[1, :] >= half_k * dilation) &
        (pose_local[1, :] < height - half_k * dilation)
    )

    # Create the array of just the optical flow windows ((C*H*W), T, V*M)
    flow_windows = np.zeros((window_size**2*2, total_keypoints))

    # Initialise points to track
    p0 = []
    skip_points = []
    for keypoint_num in range(total_keypoints):
        if valid_indices[keypoint_num]:
            x,y = pose_local[0, keypoint_num], pose_local[1, keypoint_num]
            # Create grid of positions about each keypoint ((x,y), 5, 5)
            grid = np.array(
                np.meshgrid(
                    np.linspace(x-half_k*dilation, x+half_k*dilation, window_size).int(),
                    np.linspace(y-half_k*dilation, y+half_k*dilation, window_size).int()
                )
            )
            p0.append(grid)
        else:
            # If keypoint is too close to screen edge...
            p0.append(np.zeros((2, window_size, window_size)))
            skip_points.append(keypoint_num)
            pass
    # Reshape points to track...
    p0 = rearrange(np.array(p0), 'N C H W -> (N H W) 1 C').astype('float32')

    # Estimate the optical flow (LK method)
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

    # Get vectors only for all keypoints on the frame (N=total_keypoints idk why)
    # ((N H W) C) -> ((C H W) N) equivalent to flow_window.flatten
    flow_windows = rearrange(
        (p1-p0).squeeze(),
        '(N H W) C -> (C H W) N',
        N=total_keypoints, H=window_size, W=window_size, C=2
    )
    flow_windows[:, skip_points] = np.zeros((2*(window_size**2), len(skip_points)))

    # Reshape ((C H W) (V M) -> (C H W) V M)
    # Here, C is the x and y channels of flow, H and W are height and width respectively
    flow_windows = rearrange(flow_windows, 'W (V M) -> W V M', V=17, M=2)
    return flow_windows, p0, p1


# ---------------------------------------------------------
# DRAWING TOOLS
# ---------------------------------------------------------

def draw_bones(frame, pose, person_num=None):
    frame = frame.copy()
    H,W,C = frame.shape
    pose_local = pose.detach().clone()
    pose_local[:, 0] = pose_local[:, 0] * (W-1)
    pose_local[:, 1] = pose_local[:, 1] * (H-1)
    pose_local = rearrange(pose_local, '(M V) C -> M V C', M=2, V=17)
    joint_connections = [
        [0,1], [0,2], [1,3], [2,4],
        [5,6], [5,7], [7,9], [6,8], [8,10],
        [5,11], [11,13], [13,15],
        [6,12], [12,14], [14,16]
    ]
    # Check if alpha channel exists in frame
    color = (255,0,0) if frame.shape[-1] == 3 else (255,0,0,255)
    # Get individual person's specific pose (if person_num specified)
    if person_num in [0,1]:
        pose_local = pose_local[person_num].reshape((1, 17, 2))

    for person in pose_local:
        for joint_connection in joint_connections:
            p1, p2 = joint_connection
            if person[p1, 0] <= 1.0 or person[p2, 0] <= 1.0:
                continue
            cv2.line(frame,
                    (int(person[p1,0]), int(person[p1,1])),
                    (int(person[p2,0]), int(person[p2,1])),
                    color, 3
                    )
    return frame


def draw_skel(frame, pose, person_num=None, skip_points=[], debug=False):  # Poses shape: (M V) C
    frame = frame.copy()
    H,W,C = frame.shape
    pose_local = pose.detach().clone()
    pose_local[:, 0] = pose_local[:, 0] * (W-1)
    pose_local[:, 1] = pose_local[:, 1] * (H-1)
    pose_local = rearrange(pose_local, '(M V) C -> M V C', M=2, V=17)
    if person_num != None: # If a person_num is passed, only get that specific body!
        pose_local = (pose_local[person_num]).reshape((1, 17, 2))


    inner_circ_params = {
        "radius": 5,
        "color": (0, 0, 255) if frame.shape[-1] == 3 else (0, 0, 255, 255),
        "thickness": -1
    }
    outer_circ_params = {
        "radius": 6,
        "color": (255, 0, 0) if frame.shape[-1] == 3 else (255, 0, 0, 255),
        "thickness": 3
    }
    # if frame.shape[1] < 500:
    #     pose_local[:, 0] = pose_local[:, 0] * (319 / 1919)
    #     pose_local[:, 1] = pose_local[:, 1] * (239 / 1079)
    #     circ_params = {"radius": 2, "color": (0, 0, 255), "thickness": 2}
    # Draw the skeleton keypoints on the frame
    for person in pose_local:
        for keypoint_num, keypoint in enumerate(person):
            if 0 in keypoint:
                continue
            if keypoint_num in skip_points:
                continue

            # Draw circle fill first, then the outer circle in blue
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **inner_circ_params)
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **outer_circ_params)

            if debug:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(keypoint_num), (int(keypoint[0]), int(keypoint[1])), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    return frame


def draw_flow_windows(frame, p0, p1, only_middle=False, window_size=3, mag_threshold=1000, mag_red=False):
    '''Draw optical flow windows (PoseOFF) to a frame.

    Args:
        frame (np.array): RGB video frame to draw optical flow windows to.
        p0 (np.array): Points representing the locations of pixels in PoseOFF window in frame zero.
        p1 (np.array): Estimated location of pixels tracked using LK from to frame one.
        only_middle (bool): If True, only draw the optical flow arrow centred at keypoints. Defaults to False.
        window_size (int): The width/height of optical flow window extracted using PoseOFF. Defaults to 3.
        mag_threshold (int): Optical flow vector magnitude above which will be discarded. Defaults to 1000.
        mag_red (bool): If an optical flow arrow is above mag_threshold, draw as a red circle. If False, draw nothing. Defaults to False.

    Returns:
        A numpy array with PoseOFF window optical flow arrows drawn on it.
    '''
    iterator = range((window_size**2)//2, p0.shape[0],(window_size**2)) if only_middle else range(p0.shape[0])
    arrow_color = (0, 0, 255) if frame.shape[-1] == 3 else (0, 0, 255, 255) # image transparency
    for point_num in iterator:
        mag = (
            (p1[point_num].ravel()[0]-p0[point_num].ravel()[0])**2 +
            (p1[point_num].ravel()[1]-p0[point_num].ravel()[1])**2)**(0.5)
        if mag > mag_threshold:
            if mag_red:
                frame = cv2.circle(frame, p0[point_num].ravel().astype(int), radius=1, color=(0, 0, 255), thickness=-1)
            continue
        start = p0[point_num].ravel()
        end = p1[point_num].ravel()
        frame = cv2.arrowedLine(frame, start.astype(int), end.astype(int), arrow_color, 1, tipLength=0.8)
    return frame


def draw_flow_arrows(frame, flow, step=16, scale=1.0, color=(0, 255, 0), thickness=1):
    '''Draw optical flow vectors as arrows on an image.

    Args:
        frame (np.array): BGR image (H x W x 3) to draw on (will be copied).
        flow (np.array): Optical flow array (H x W x 2).
        step (int): Grid spacing in pixels — controls how many arrows are drawn.
        scale (float): Multiplier for arrow length (useful if flow magnitudes are tiny/huge).
        color (tuple[int]): Arrow color as (B, G, R).
        thickness (int): Arrow line thickness in pixels.

    Returns:
        A copy of the image with arrows drawn on it.
    '''
    if frame.ndim == 2:
        out = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        out = frame.copy()

    h, w = out.shape[:2]

    # Build a regular grid of sample points
    xs = np.arange(step // 2, w, step)
    ys = np.arange(step // 2, h, step)
    xv, yv = np.meshgrid(xs, ys)          # both shape: (len(ys), len(xs))

    # Sample the flow at every grid point
    fx = flow[yv, xv, 0] * scale          # horizontal displacement
    fy = flow[yv, xv, 1] * scale          # vertical displacement

    # Arrow tip coordinates
    x_end = (xv + fx).astype(int)
    y_end = (yv + fy).astype(int)

    # Draw each arrow
    for (x0, y0, x1, y1) in zip(xv.ravel(), yv.ravel(), x_end.ravel(), y_end.ravel()):
        cv2.arrowedLine(out, (x0, y0), (x1, y1), color, thickness, tipLength=0.3)

    return out


def draw_flow_hsv(frame, flow, norm=True):
    '''TODO: Docstring

    Args:
    ...
    norm (bool): Normalise the HSV values stretching them between 0 and 255. Default is True.
    '''
    if frame.ndim == 2:
        out = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        out = frame.copy()

    # Create a mask we will use to generate HSV color img
    hsv_mask = np.zeros_like(out)
    hsv_mask[..., 1] = 255 # set channel (saturation) to max
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # convert flow vecs to polar coords

    # Convert angles from degrees to radians
    hsv_mask[..., 0] = ang*180/np.pi/2
    if norm:
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255).astype(np.uint8)
    # Take the hsv mask and convert it into BGR color space...
    out = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    return out
