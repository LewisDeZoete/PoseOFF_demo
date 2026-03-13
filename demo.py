#!/usr/bin/env python3

import argparse
import cv2
from utils import *

def get_args():
    parser = argparse.ArgumentParser(
        prog="PoseOFF_feature_demo",
        description="Demonstration of lightweight PoseOFF feature extraction method, using YOLO pose and LK optical flow estimation. \nPRESS Q TO CLOSE WINDOW.",
    )
    parser.add_argument('-t', '--threshold', default=0.2,
                        help="Confidence threshold below which pose keypoints will be discarded, between 0.0 and 1.0 (default: 0.2).")
    parser.add_argument('-w', '--window_size', default=5,
                        help="Width of square optical flow sampling window - must be an odd number - for example window_size=5 would result in a 5*5 pixel window (default: 5)")
    parser.add_argument('-d', '--dilation', default=3,
                        help="Dilation factor of sampling window, a higher dilation means a more spread sampling window (default: 3)")
    parser.add_argument('-c', '--camera_number', default=0,
                        help="Camera number to stream from, this may require some trial and error... (default: 0)")
    parser.add_argument('-v', '--video_path',
                        help="If not using live webcam, pass the video_path!")
    parser.add_argument('-x', '--first_x', default=0,
                        help="If -v (video path) is passed, dictates how many frames to move ahead to begin calculating diference images.")
    parser.add_argument('-m', '--only_middle', action='store_true',
                        help="If passed, only draw the middle optical flow arrow on each pose keypoint - store_true (default: False)")
    args = parser.parse_args()

    # Checking input values...
    assert 0 < float(args.threshold) < 1, "--threshold must be a float between 0.0 and 1.0!"
    assert int(args.window_size) % 2 == 1, "Window size must be an odd number (so it can be centred of a pose keypoint...)"
    assert int(args.dilation) > 1, "Dilation factor must be greater than 1..."
    if args.first_x:
        try:
            assert int(args.first_x) > 0, "First_x must be greater than 0..."
        except ValueError:
            print("Please put in an integer for '--first_x'")

    # Convert to correct datatype...
    args.threshold = float(args.threshold)
    args.window_size = int(args.window_size)
    args.dilation = int(args.dilation)
    args.camera_number = int(args.camera_number)

    return args


def main(args, pose_model):
    '''Main loop for PoseOFF feature extraction and visualisation.

    Args:
        args (argparse.Namespace): argparse object containing variables for threshold, window_size, dilation, camera_number and optional only_middle argument.
        pose_model (ultralytics.models.yolo.model.YOLO): Initialised pre-trained YOLO Pose model.
    '''
    print("\n ------- PRESS `Q` TO QUIT ------ \n")
    cap = cv2.VideoCapture(args.camera_number)
    ret, img1 = cap.read()
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im_height, im_width = img1_grey.shape

    while cap.isOpened():
        ret, img2 = cap.read()
        if not ret:
            print("Can't open frame")
            break
        # Get the poses using YOLO
        poses = get_poses(img2, pose_model, threshold=args.threshold)

        # Convert the frame to grey to prep for LK flow estimation
        img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate PoseOFF windows using LK flow
        poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=args.window_size, dilation=args.dilation)

        # Drawing utilities
        img2 = draw_bones(img2, poses)
        # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
        img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size)

        # Resize the input image...
        img2 = cv2.resize(img2, (im_width*2, im_height*2))

        # Show the frame
        cv2.imshow('Frame', img2)
        if cv2.waitKey(1) == ord('q'):
            break

        # Set the current frame to the old frame before retrieving a new one...
        img1_grey = img2_grey.copy()
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def video(args, pose_model):
    # TODO: fold this in to main...
    # Get the first frame
    cap = cv2.VideoCapture(args.video_path)
    ret, img1 = cap.read()
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im_height, im_width = img1_grey.shape

    # first_x dictates how many frames to move ahead to get difference images
    if int(args.first_x) > 0:
        for i in range(int(args.first_x)):
            ret, img2 = cap.read()

        # Convert second image to grey...
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Get the poses using YOLO
        poses = get_poses(img2, pose_model, threshold=args.threshold)

        # Calculate PoseOFF windows using LK flow
        poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=args.window_size, dilation=args.dilation)

        # Drawing utilities
        img2 = draw_bones(img2, poses)
        # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
        img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size)

        # Resize the input image...
        img2 = cv2.resize(img2, (im_width*2, im_height*2))

        # Show the frame
        cv2.imshow('Frame', img2)
        if cv2.waitKey(1) == ord('q'):
            quit()

        # Set the current frame to the old frame before retrieving a new one...
        img1_grey = img2_grey.copy()

    # If first_x isn't passed, just show video
    if not args.first_x:
        while cap.isOpened():
            ret, img2 = cap.read()
            if not ret:
                print("Can't open frame")
                break
            # Get the poses using YOLO
            poses = get_poses(img2, pose_model, threshold=args.threshold)

            # Convert the frame to grey to prep for LK flow estimation
            img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate PoseOFF windows using LK flow
            poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=args.window_size, dilation=args.dilation)

            # Drawing utilities
            img2 = draw_bones(img2, poses)
            # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
            img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size, mag_threshold=1000)

            # Resize the input image...
            img2 = cv2.resize(img2, (im_width*2, im_height*2))

            # Show the frame
            cv2.imshow('Frame', img2)
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite("TMP.png", img2)
                break

            # Set the current frame to the old frame before retrieving a new one...
            img1_grey = img2_grey.copy()

if __name__ == '__main__':
    # Parse command line arguments
    args = get_args()
    # Create YOLO-pose model
    pose_model = YOLO("yolo11m-pose.pt")

    # If a video path is passed, use the offline method
    if args.video_path is not None:
        video(args, pose_model=pose_model)
    else:
        # Otherwise, run main script
        main(args, pose_model=pose_model)
