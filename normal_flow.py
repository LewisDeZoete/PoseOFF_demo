#!/usr/bin/env python3

import os
import os.path as osp
import cv2
from utils import *
import argparse

class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="PoseOFF_feature_demo",
            description="Demonstration of lightweight PoseOFF feature extraction method, using YOLO pose and LK optical flow estimation. \nPRESS Q TO CLOSE WINDOW.",
        )
        parser.add_argument('--input_type', default='camera',
                            help="Input type must be in ['camera', 'video', 'frames'].\n"
                            "if 'camera', -c --camera_number must be set to an appropriate number (default 0).\n"
                            "(default: 'camera')")
        parser.add_argument('--input_path',
                            help="Path to video of folder containing frames.")
        parser.add_argument('--temporal_type', default='5pt',
                            help="Type of temporal gradient calculation.")
        parser.add_argument('--resize', default=1,
                            help="Multiplier for resizing the output image.")
        parser.add_argument('--ksize', default=9,
                            help="Gaussian blur kernel size.")
        parser.add_argument('--HSV', action='store_true',
                            help="Display as HSV image rather than flow arrows")
        parser.add_argument('--write_video', action='store_true',
                            help="If passed, writes a video to the 'output/' folder")
        
        parsed = parser.parse_args()

        assert parsed.temporal_type in ["5pt", "sobel"], "temporal type must be '5pt' or 'sobel'."
        assert parsed.ksize%2 == 1, "Gaussian kernel size must be an odd integer."

        self.input_type = parsed.input_type
        self.input_path = parsed.input_path if self.input_type in ['video', 'frames'] else 0
        self.temporal_type = parsed.temporal_type
        self.resize = parsed.resize
        self.HSV = parsed.HSV
        self.write_video = parsed.write_video


def main():
    # Parse the commandline arguments!
    args = Args()
    
    print("\n ------- PRESS `Q` TO QUIT ------ \n")
    cap = cv2.VideoCapture(args.input_path)
    ret, img1 = cap.read()
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im_height, im_width = img1_grey.shape

    # Create the 5 frame estimator
    estimator = NormalFlowEstimator(
        buffer_size=5,
        grad_thresholds=[0.5, 500.0],
        temporal_estimator=temporal_gradient_5point if args.temporal_type == '5pt' \
        else temporal_gradient_sobel,
        ksize=13
    )

    # If write_video==True, write video with the same name as the input
    if args.write_video:
        os.makedirs("output", exist_ok=True) # Create output folder
        video_filename = osp.join(
            './output',
            osp.basename(str(args.input_path)).split('.')[0] + ".avi"
        )
        FPS = cap.get(cv2.CAP_PROP_FPS)
        W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), FPS, (W, H))
        print(f"Writing video to {video_filename}")

    # Frame counter for saving video frames
    frame_count = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Ran out of frames!")
            break
        # img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate normal flow!
        norm_flow, frame = estimator.push(frame)

        if norm_flow is not None:
            frame = draw_flow_hsv(frame, norm_flow) if args.HSV else \
                draw_flow_arrows(
                    frame,
                    -1*norm_flow.astype(np.float32),
                    step=20,
                    thickness=2
                )
        else:
            frame_count+=1
            continue

        # If write_video, don't show the frame, only write it!
        if args.write_video:
            out.write(frame)
        else:
            # Resize the input image...
            frame = cv2.resize(frame, (int(im_width * args.resize), int(im_height * args.resize)))

            # Show the frame
            cv2.imshow('Frame', frame)
            keypress = cv2.waitKey(1)
            if keypress == ord('q'):
                print("Exiting video playback")
                break
            elif keypress == ord("w"):
                img_filename = osp.join(
                    './output',
                    osp.basename(str(args.input_path)).split('.')[0] + ".png"
                )
                print(f"Writing: {img_filename}")
                cv2.imwrite(img_filename, frame)

        # img1_grey = img2_grey.copy()
        frame_count+=1

    cap.release()
    if args.write_video:
        out.release()


if __name__=="__main__":
    # TODO: Gaussian smoothing
    main()
        
