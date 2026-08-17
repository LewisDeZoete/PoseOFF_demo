#!/usr/bin/env python3
import os.path as osp
import os
import re
import cv2
import numpy as np
from utils import get_poses, draw_bones, draw_skel
import torch
from einops import rearrange
from ultralytics import YOLO

vid = "shake" # punch / shake

def main(vid):
    rgb_path = f'input/punch_shake/C{vid}_color.mp4'
    poses_path = f'input/punch_shake/{vid}-poses.npy'
    flow_path = f'input/punch_shake/{vid}-flow.npy'

    poses = torch.from_numpy( np.load(poses_path) )
    flows = torch.from_numpy( np.load(flow_path) )

    flows = rearrange(flows, 'T C H W -> T H W C')

    print(f'Flow shape: {flows.shape}')
    print(f'Pose shape: {poses.shape}')

    cap = cv2.VideoCapture(rgb_path)
    frame_no = 0


    while cap.isOpened():
        ret, frame = cap.read()
        frame = np.zeros((480, 640, 4))
        if not ret:
            print("Can't open frame")
            break

        frame = draw_bones(frame, poses[frame_no])
        frame = draw_skel(frame, poses[frame_no])

        cv2.imshow('NE', frame)
        if cv2.waitKey(0) == ord('q'):
            break

        frame_no +=1

    cap.release()
    import cv2

def write_video(frame_paths, output_path, fps=30, background=(0, 0, 0)):
    """
    Write a list of image files to a single video file.

    Args:
        frame_paths: list of file paths (in order) to the frames.
        output_path: path to write the output video (e.g. "out.avi").
        fps: frames per second.
        background: RGB color to flatten alpha channel onto, if present.
    """
    print(f"Writing video to {output_path}")
    if not frame_paths:
        raise ValueError("frame_paths is empty")

    # Read first frame to get dimensions
    first = cv2.imread(frame_paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Could not read {frame_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Could not read {path}")

        # Flatten alpha channel if present
        if frame.shape[-1] == 4:
            bgr = frame[..., :3].astype(np.float32)
            alpha = (frame[..., 3:4].astype(np.float32)) / 255.0
            bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
            frame = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', s)]
if __name__=="__main__":
    vid = "shake"
    type = "flow"
    frame_names = sorted(os.listdir(f'input/punch_shake/frames/{vid}/{type}'), key=natural_sort_key)
    frame_paths = [f'input/punch_shake/frames/{vid}/{type}/{frame_name}' for frame_name in frame_names]
    write_video(frame_paths, f'./input/punch_shake/{vid}-{type}.mp4')
    # main(vid)
