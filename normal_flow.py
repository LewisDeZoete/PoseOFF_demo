#!/usr/bin/env python3

import os
import os.path as osp
import cv2
from utils import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        prog="PoseOFF_feature_demo",
        description="Demonstration of lightweight PoseOFF feature extraction method, using YOLO pose and LK optical flow estimation. \nPRESS Q TO CLOSE WINDOW.",
    )
    parser.add_argument('--input_type', default='camera',
                        help="Input type must be in ['camera', 'video', 'frames'].\n"
                        "if 'camera', -c --camera_number must be set to an appropriate number (default 0).\n"
                        "(default: 'camera')")
    parser.add_argument('--temporal_type', default='5pt',
                        help="Type of temporal gradient calculation")
    parser.add_argument('--HSV', action='store_true',
                        help="Display as HSV image rather than flow arrows")
    parser.add_argument('--write_video', action='store_true',
                        help="If passed, writes a video to the 'output/' folder")
    args = parser.parse_args()

    assert args.temporal_type in ["5pt", "sobel"], "temporal type must be '5pt' or 'sobel'"

    return args


def normal_flow_frames(
        input_path,
        five_frame=False,
        HSV=False,
        write_video:bool=False,
        resize:int=1,
):
    '''TODO: Move this to a separate file!'''
    print("\n ------- PRESS `Q` TO QUIT ------ \n")
    cap = cv2.VideoCapture(input_path)
    ret, img1 = cap.read()
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im_height, im_width = img1_grey.shape

    # Create the 5 frame estimator
    estimator = NormalFlowEstimator(
        buffer_size=5,
        grad_thresholds=[0.5, 500.0],
        temporal_estimator=temporal_gradient_5point if args.temporal_type == '5pt' \
        else temporal_gradient_sobel
    )

    # multiplier (5 frame normals are backwards for some reason?)
    vector_multi = -1 if five_frame else 1

    # If write_video==True, write video with the same name as the input
    if write_video:
        os.makedirs("output", exist_ok=True) # Create output folder
        video_filename = osp.join(
            './output',
            osp.basename(input_path).split('.')[0] + \
            f"{'5' if five_frame else '2'}.avi"
        )
        FPS = cap.get(cv2.CAP_PROP_FPS)
        W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), FPS, (W, H))
        print(f"Writing video to {video_filename}")

    # Frame counter for saving video frames
    frame_count = 0
    while cap.isOpened():

        ret, img2 = cap.read()
        raw_img = img2.copy()
        if not ret:
            print("Ran out of frames!")
            break
        img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate either two or five frame normal flow
        if five_frame:
            norm_flow, img2 = estimator.push(img2)
        else:
            norm_flow = get_norm_flows(img1_grey, img2_grey)

        if norm_flow is not None:
            img2 = draw_flow_hsv(img2, norm_flow) if HSV else \
                draw_flow_arrows(
                    img2,
                    vector_multi*norm_flow.astype(np.float32),
                    step=20,
                    thickness=2
                )
        else:
            frame_count+=1
            continue

        # If write_video, don't show the frame, only write it!
        if write_video:
            out.write(img2)
        else:
            # Resize the input image...
            img2 = cv2.resize(img2, (int(im_width * resize), int(im_height * resize)))

            # Show the frame
            cv2.imshow('Frame', img2)
            keypress = cv2.waitKey(0)
            if keypress == ord('q'):
                print("Exiting video playback")
                break
            elif keypress == ord("w"):
                img_filename = osp.join(
                    './output',
                    osp.basename(input_path).split('.')[0] + \
                    f"_{'5' if five_frame else '2'}frame_{frame_count}.png"
                )
                print(f"Writing: {img_filename}")
                cv2.imwrite(img_filename, raw_img)

        img1_grey = img2_grey.copy()
        frame_count+=1

    cap.release()
    if write_video:
        out.release()


if __name__=="__main__":
    args = get_args()
    # TODO: Gaussian smoothing
