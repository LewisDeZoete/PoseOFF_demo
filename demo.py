#!/usr/bin/env python3

import os
import os.path as osp
import argparse
import cv2
from utils import *

def get_args():
    parser = argparse.ArgumentParser(
        prog="PoseOFF_feature_demo",
        description="Demonstration of lightweight PoseOFF feature extraction method, using YOLO pose and LK optical flow estimation. \nPRESS Q TO CLOSE WINDOW.",
    )
    parser.add_argument('--input_type', default='camera',
                        help="Input type must be in ['camera', 'video', 'frames'].\n"
                        "if 'camera', -c --camera_number must be set to an appropriate number (default 0).\n"
                        "If 'video', -i --input_path must be set to a video path (e.g. '.mp4').\n"
                        "If 'frames', -i --input_path must be set to a folder containing video frames.\n"
                        "(default: 'camera')")
    parser.add_argument('-t', '--threshold', default=0.2,
                        help="Confidence threshold below which pose keypoints will be discarded, between 0.0 and 1.0 (default: 0.2).")
    parser.add_argument('-w', '--window_size', default=5,
                        help="Width of square optical flow sampling window - must be an odd number - for example window_size=5 would result in a 5*5 pixel window (default: 5)")
    parser.add_argument('-d', '--dilation', default=3,
                        help="Dilation factor of sampling window, a higher dilation means a more spread sampling window (default: 3)")
    parser.add_argument('-m', '--mag_threshold', default=100,
                        help="Optical flow magnitude threshold, limiting how large flow arrows will be (default: 100)")
    parser.add_argument('-c', '--camera_number', default=0,
                        help="Camera number to stream from, this may require some trial and error... (default: 0)")
    parser.add_argument('-i', '--input_path',
                        help="If not using live webcam, pass the input path for videos or frames.")
    parser.add_argument('-s', '--skip_frames', default=0,
                        help="Number of video frames in between each PoseOFF extimation - effectively controls framerate. A higher number means more frames are skipped (default: 0)")
    parser.add_argument('-o', '--only_middle', action='store_true',
                        help="If passed, only draw the middle optical flow arrow on each pose keypoint - store_true (default: False)")
    parser.add_argument('-r', '--resize', default=1,
                        help="Amount to resize the output video by (default: 1)")
    parser.add_argument('--write_video', action='store_true',
                        help="If passed, writes a video to the 'output/' folder")
    args = parser.parse_args()

    # Checking input values...
    assert 0 < float(args.threshold) < 1, "--threshold must be a float between 0.0 and 1.0!"
    assert int(args.window_size) % 2 == 1, "Window size must be an odd number (so it can be centred of a pose keypoint.)"
    assert int(args.dilation) > 1, "Dilation factor must be greater than 1."
    assert 0 < int(args.mag_threshold) < 1920, "Magnitude threshold must be between 1 and 1920."

    # Check the input_type and associated variables are passed
    assert args.input_type in ['camera', 'video', 'frames'], "--input_type must be one of 'camera', 'video', 'frames'"
    if args.input_type == 'camera':
        try:
            assert int(args.camera_number) >= 0, "Camera number must be >= zero."
        except ValueError:
            print("Camera number must be an integer")
    elif args.input_type == 'video':
        assert osp.isfile(args.input_path), "For input_type = video, input_path must be a path to a file."
        assert float(args.resize) > 0, "The resize argument must be a non-zero integer."
        print(f"Running demo for video: {args.input_type}")
    elif args.input_type == 'frames':
        assert osp.isdir(args.input_path), "For input_type = frames, input_path must be a folder."
        assert len(os.listdir(input_path)) > 0, "input_path must point to a folder containing frames."
        print(f"Running demo for {len(os.listdir(input_path))} frames.")

    # Convert to correct datatype...
    args.threshold = float(args.threshold)
    args.window_size = int(args.window_size)
    args.dilation = int(args.dilation)
    args.mag_threshold = int(args.mag_threshold)
    args.camera_number = int(args.camera_number)
    args.resize = float(args.resize)

    # Ensure if skip_frames is passed, it's a non-zero int
    if args.skip_frames:
        try:
            assert int(args.skip_frames) > 0, "skip_frames must be greater than 0..."
        except ValueError:
            print("Please put in an integer for '--skip_frames'")

    return args


class Main:
    '''TODO: docstring'''
    def __init__(self, args, pose_model):
        self.args = args
        self.pose_model = pose_model

        if args.input_type == 'camera':
            self.camera()
        elif args.input_type == 'video':
            self.video()
        elif args.input_type == 'frames':
            self.frames()
    def camera(self):
        print("\n ------- PRESS `Q` TO QUIT ------ \n")
        cap = cv2.VideoCapture(self.args.camera_number)
        ret, img1 = cap.read()
        img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        im_height, im_width = img1_grey.shape

        while cap.isOpened():
            ret, img2 = cap.read()
            if not ret:
                print("Can't open frame")
                break
            # Get the poses using YOLO
            poses = get_poses(img2, pose_model, threshold=self.args.threshold)

            # Convert the frame to grey to prep for LK flow estimation
            img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate PoseOFF windows using LK flow
            poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=self.args.window_size, dilation=self.args.dilation)

            # Drawing utilities
            img2 = draw_bones(img2, poses)
            # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
            img2 = draw_flow_windows(img2, p0, p1, only_middle=self.args.only_middle, window_size=self.args.window_size, mag_threshold=self.args.mag_threshold)

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

    def video(self):
        print("\n ------- PRESS `Q` TO QUIT ------ \n")
        cap = cv2.VideoCapture(self.args.input_path)
        ret, img1 = cap.read()
        img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        im_height, im_width = img1_grey.shape

        # If write_video==True, write video with the same name as the input
        if self.args.write_video:
            os.makedirs("output", exist_ok=True) # Create output folder
            video_filename = osp.join(
                './output',
                osp.basename(self.args.input_path).split('.')[0] + ".avi"
            )
            FPS = cap.get(cv2.CAP_PROP_FPS)
            W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), FPS, (W, H))
            print(f"Writing video to {video_filename}")

        while cap.isOpened():
            # Skip_frames frames if passed as argument
            if int(self.args.skip_frames) > 0:
                for i in range(int(self.args.skip_frames)):
                    ret, img2 = cap.read()
                    if not ret:
                        print("Ran out of frames...")
                        break

            # Read the next frame (quit if end of video)
            ret, img2 = cap.read()
            if not ret:
                print("Can't open frame")
                break
            # Get the poses using YOLO
            poses = get_poses(img2, pose_model, threshold=self.args.threshold)

            # Convert the frame to grey to prep for LK flow estimation
            img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate PoseOFF windows using LK flow
            poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=self.args.window_size, dilation=self.args.dilation)

            # Drawing utilities
            img2 = draw_bones(img2, poses)
            # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
            img2 = draw_flow_windows(img2, p0, p1, only_middle=self.args.only_middle, window_size=self.args.window_size, mag_threshold=self.args.mag_threshold)

            # If write_video, don't show the frame, only write it!
            if self.args.write_video:
                out.write(img2)
            else:
                # Resize the input image...
                img2 = cv2.resize(img2, (int(im_width * self.args.resize), int(im_height * self.args.resize)))

                # Show the frame
                cv2.imshow('Frame', img2)
                if cv2.waitKey(1) == ord('q'):
                    print("Exiting video playback")
                    break

            # Set the current frame to the old frame before retrieving a new one...
            img1_grey = img2_grey.copy()

        cap.release()
        if self.args.write_video:
            out.release()

    def frames(self):
        pass

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
        img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size, mag_threshold=args.mag_threshold)

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

    # skip_frames dictates how many frames to move ahead to get difference images
    if int(args.skip_frames) > 0:
        for i in range(int(args.skip_frames)):
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
        img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size, mag_threshold=args.mag_threshold)

        # Resize the input image...
        img2 = cv2.resize(img2, (im_width*2, im_height*2))

        # Show the frame
        cv2.imshow('Frame', img2)
        if cv2.waitKey(1) == ord('q'):
            quit()

        # Set the current frame to the old frame before retrieving a new one...
        img1_grey = img2_grey.copy()

    # If skip_frames isn't passed, just show video
    if not args.skip_frames:
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
            img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size, mag_threshold=args.mag_threshold)

            # Resize the input image...
            img2 = cv2.resize(img2, (im_width*2, im_height*2))

            # Show the frame
            cv2.imshow('Frame', img2)
            if cv2.waitKey(1) == ord('q'):
                cv2.imwrite("TMP.png", img2)
                break

            # Set the current frame to the old frame before retrieving a new one...
            img1_grey = img2_grey.copy()


def frames(args, pose_model, class_files, save_dir):
    for class_name, files in class_files.items():
        for i in range(len(files)-1):
            print(f"{files[i]}\n{files[i+1]}\n\n")
            img1 = cv2.imread(files[i])
            img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(files[i+1])
            img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # Get the poses using YOLO
            poses = get_poses(img2, pose_model, threshold=args.threshold)
            # Calculate PoseOFF windows using LK flow
            poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=args.window_size, dilation=args.dilation)

            # Drawing utilities
            img2 = draw_bones(img2, poses)
            # img2 = np.zeros((1080, 1920, 4))
            # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
            img2 = draw_flow_windows(img2, p0, p1, only_middle=args.only_middle, window_size=args.window_size, mag_threshold=args.mag_threshold, mag_red=True)

            save_name = files[i].split('\\')[-1].split('.')[0] + '-' + \
            files[i+1].split('-')[-1].split('.')[0].split('f')[-1] + '.png'
            cv2.imwrite(osp.join(save_dir, save_name), img2)
            # cv2.imshow("Frame", img2)
            # keypress = cv2.waitKey(0)
            # if keypress == ord('q'):
            #     quit()
            # elif keypress == ord('s'):
            #     # Incredibly cursed...
            #     save_name = files[i].split('\\')[-1].split('.')[0] + '-' + \
            #     files[i+1].split('-')[-1].split('.')[0].split('f')[-1] + '.png'
            #     print(f"SAVING: {save_name}")
            #     cv2.imwrite(osp.join(save_dir, save_name), img2)


def pose_frames(args, pose_model, class_files, save_dir):
    save_dir = osp.join(save_dir, "Pose")
    os.makedirs(save_dir, exist_ok=True)
    for class_name, files in class_files.items():
        for filename in files:
            img = cv2.imread(filename)
            poses = get_poses(img, pose_model, threshold=args.threshold)
            img = np.zeros((1080, 1920, 4))
            img = draw_bones(img, poses)
            img = draw_skel(img, poses) # Uncomment this to draw the skeleton joint
            print(f"Saving {filename.split("\\")[-1]} to {osp.join(save_dir, filename.split("\\")[-1])}")
            cv2.imwrite(osp.join(save_dir, filename.split("\\")[-1]), img)


def crop_large_imgs(in_path="./TMP_SAVE/PoseOFF", x_origin=250, y_origin=150, size=900):
    cropped_img_path = osp.join(in_path, 'cropped')
    # Make the cropped image path if it doesn't exist
    os.makedirs(cropped_img_path, exist_ok=True)

    img_names = os.listdir(in_path)
    for img_name in img_names:
        if not osp.isfile(osp.join(in_path, img_name)):
            continue
        print(f"Cropping: {img_name}")
        # if not img_name[:4] == "POSE": # Don't crop pose diagrams drawn with matplotlib...
        img_in_path = osp.join(in_path, img_name)
        img_out_path = osp.join(in_path, 'cropped', img_name)

        # Read the image
        img = cv2.imread(img_in_path, cv2.IMREAD_UNCHANGED)
        if img_name[:2] == "CV":
            print(img.shape)
        cropped = img[y_origin:y_origin+size, x_origin:x_origin+size]

        cv2.imwrite(img_out_path, cropped)


if __name__ == '__main__':
    # Parse command line arguments
    args = get_args()
    # Create YOLO-pose model
    pose_model = YOLO("yolo11m-pose.pt")

    Main(args, pose_model)
    cv2.destroyAllWindows()
    # # If a video path is passed, use the offline method
    # if args.video_path is not None:
    #     video(args, pose_model=pose_model)
    # else:
    #     # Otherwise, run main script
    #     main(args, pose_model=pose_model)


    # import os
    # import os.path as osp

    # class_files = {}
    # # for class_name in os.listdir("./class_examples/"):
    # #     class_files[class_name] = [
    # #         osp.join("./class_examples/", class_name, filename)
    # #         for filename in os.listdir(osp.join("./class_examples", class_name))
    # #     ]
    # #
    # #
    # #
    # # TODO: Give the option to choose a specific class to calc PoseOFF for!
    # class_name = "71-make_ok_sign" # 6, 27, 43, 95, 98, 113,
    # data_type = "PoseOFF" # Pose, Flow, PoseOFF, RGB
    # save_dir = f"./TMP_SAVE/{data_type}/{class_name}"
    # os.makedirs(save_dir, exist_ok=True)

    # file_names = os.listdir(osp.join("./class_examples", class_name))
    # # Sort correctly by frame numbers...
    # sorted_filenames = sorted(file_names, key=lambda x: int(x.split('-f')[-1].split('.')[0]))
    # class_files[class_name] = [
    #     osp.join("./class_examples/", class_name, filename)
    #     for filename in sorted_filenames
    # ]
    # # if data_type == "Pose":
    # #     pose_frames(args, pose_model, class_files, save_dir)
    # # else:
    # #     frames(args, pose_model, class_files, save_dir)


    # crop_large_imgs(
    #     in_path=f"./TMP_SAVE/{data_type}/{class_name}",
    #     x_origin=450,
    #     y_origin=250,
    #     size=900,
    # )
