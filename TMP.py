#!/usr/bin/env python3
import cv2
import numpy as np

vid = "punch" # punch / shake

def main(vid):
    vid_path = f'input/punch_shake/{vid}-rgb.npy'
    vid_np = np.load(vid_path).transpose(0, 2,3,1)

    print(vid_np.shape)
    while True:
        cv2.imshow('NE', vid_np[0].astype(np.float32))
        if cv2.waitKey(1) == ord('q'):
            break
    # print("\n ------- PRESS `Q` TO QUIT ------ \n")
    # ret, img1 = cap.read()
    # img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # im_height, im_width = img1_grey.shape

    # while cap.isOpened():
    #     ret, img2 = cap.read()
    #     if not ret:
    #         print("Can't open frame")
    #         break
    #     # # Get the poses using YOLO
    #     # poses = get_poses(img2, self.pose_model, threshold=self.args.threshold)

    #     # # Convert the frame to grey to prep for LK flow estimation
    #     # img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #     # # Calculate PoseOFF windows using LK flow
    #     # poseoff, p0, p1 = poseoff_lk(img1_grey, img2_grey, poses, window_size=self.args.window_size, dilation=self.args.dilation)

    #     # # Drawing utilities
    #     # img2 = draw_bones(img2, poses)
    #     # # img2 = draw_skel(img2, poses) # Uncomment this to draw the skeleton joint
    #     # img2 = draw_flow_windows(img2, p0, p1, only_middle=self.args.only_middle, window_size=self.args.window_size, mag_threshold=self.args.mag_threshold)

    #     # Resize the input image...
    #     img2 = cv2.resize(img2, (im_width*2, im_height*2))

    #     # Show the frame
    #     cv2.imshow('Frame', img2)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    #     # Set the current frame to the old frame before retrieving a new one...
    #     img1_grey = img2_grey.copy()

    # # Cleanup
    # cap.release()
    # cv2.destroyAllWindows()

if __name__=="__main__":
    main(vid)
