#!/usr/bin/env python3
import os.path as osp
import os
import re
import cv2
import numpy as np
from utils import draw_flow_arrows, get_poses, draw_bones, draw_skel
import torch
from einops import rearrange
from ultralytics import YOLO

vid = "shake" # punch / shake

def main(vid):
    rgb_path = f'input/punch_shake/C{vid}_color.mp4'
    poses_path = f'input/punch_shake/{vid}-poses.npy'
    flow_path = f'input/punch_shake/{vid}-flow.npy'

    poses = np.load(poses_path)
    flows = np.load(flow_path)

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

        frame = draw_flow_arrows(frame, flows[frame_no], step=11, scale=2.5, color=(0,255,0,0))

        cv2.imshow('NE', frame)
        if cv2.waitKey(0) == ord('q'):
            break

        frame_no +=1

    cap.release()

def write_frames(vid):
    outdir = f"output/punch_shake/{vid}"
    os.makedirs(osp.join(outdir,"poses"), exist_ok=True)
    os.makedirs(osp.join(outdir,"flows"), exist_ok=True)
    poses_path = f'input/punch_shake/{vid}-poses.npy'
    flows_path = f'input/punch_shake/{vid}-flows.npy'

    poses = np.load(poses_path) # T (M V) C
    flows = np.load(flows_path)
    flows = rearrange(flows, 'T C H W -> T H W C')

    _, H, W, _ = flows.shape

    for frame_no, flow in enumerate(flows):
        # Create and draw the pose frame!
        pose_frame = np.zeros((H, W, 4))
        pose_frame = draw_bones(pose_frame, poses[frame_no])
        pose_frame = draw_skel(pose_frame, poses[frame_no])
        cv2.imwrite(osp.join(outdir, "poses", f"frame-{frame_no}.png"), pose_frame)

        flow_frame = np.zeros((H, W, 4))
        flow_frame = draw_flow_arrows(flow_frame, flow, step=11, scale=2.5, color=(0,0,255,255), thickness=2)
        cv2.imwrite(osp.join(outdir, "flows", f"frame-{frame_no}.png"), flow_frame)



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', s)]

if __name__=="__main__":
    vid = "punch"

    # main(vid)
    write_frames(vid)
