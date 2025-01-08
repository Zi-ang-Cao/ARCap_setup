"""
Collect ARCap data by wired connecting the ZED camera to the Ubuntu and wireless connecting the Quest3 to the same PC.

Author: @Zi-ang-Cao
Date: Jan 2025

Usage:
    python zed_data_collection_server.py

How to use no_glove left hand to collect the left hand data?
    1. run `python zed_data_collection_server.py` on Ubuntu
    2. Open the ARCap app on the Quest3
    3. Type the IP address of the Ubuntu PC Machine in the ARCap app
    4. Use controller to refine the starting position of the vitural robot
    5. Double tap two controllers onto each other and switch to from using controller to the hands
    6. Align the left hand with the virtual robot, and then open-close the left hand to simulate pressing "x" to finish the initialization.
    7. Use the right hand to label starting and ending recording.

"""

import socket
import time
from argparse import ArgumentParser
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb
from rigidbodySento import create_primitive_shape
from ip_config import (
    VR_HOST,
    LOCAL_HOST,
    IK_RESULT_PORT, 
    POSE_CMD_PORT,
)
USE_ZED = True
if USE_ZED:
    from zed_module import ZEDCameraModule
else:
    from realsense_module import DepthCameraModule
from quest_robot_module import QuestLeftArmGripperNoRokokoModule
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--handedness", type=str, default="left")
    parser.add_argument("--exp_name", type=str, default="debug_left_wrist_offset_Jan7")
    parser.add_argument("--no_camera", action="store_true", default=False)
    args = parser.parse_args()
    
    c = pb.connect(pb.DIRECT)
    vis_sp = []
    c_code = c_code = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]
    for i in range(4):
        vis_sp.append(create_primitive_shape(pb, 0.1, pb.GEOM_SPHERE, [0.02], color=c_code[i]))
    
    if not args.no_camera:
        if USE_ZED:
            camera = ZEDCameraModule(is_decimate=False, visualize=False)
        else:
            camera = DepthCameraModule(is_decimate=False, visualize=False)
    if args.handedness == "right":
        raise NotImplementedError   # There is no right arm leap hand module without rokoko yet.
    else:
        quest = QuestLeftArmGripperNoRokokoModule(VR_HOST, LOCAL_HOST, POSE_CMD_PORT, IK_RESULT_PORT, vis_sp=vis_sp)

    start_time = time.time()
    fps_counter = 0
    packet_counter = 0
    print("Initialization completed")
    current_ts = time.time()
    running = True
    debug_print = True
    while running:
        now = time.time()
        # TODO: May cause communication issues, need to tune on AR side.
        if now - current_ts < 1 / args.frequency: 
            continue
        else:
            current_ts = now
        try:
            if not args.no_camera:
                point_cloud = camera.receive()
                if debug_print:
                    print(point_cloud.shape)
                    debug_print = False
            # Directly get the wrist position and orientation from the Meta Quest3.
            wrist, head_pose= quest.receive()
            if wrist is not None:
                wrist_orn = Rotation.from_quat(wrist[1])
                wrist_pos = wrist[0]
                head_pos = head_pose[0]
                head_orn = Rotation.from_quat(head_pose[1])
                # Only back to the time of using rokoko, we need to find the hand_tip_pose
                # Otherwise, we directly use the quest hand pose (and set hand_tip_pose = None)
                hand_tip_pose = None
                arm_q, hand_q, wrist_pos, wrist_orn = quest.solve_system_world(wrist_pos, wrist_orn, hand_tip_pose)
                action = quest.send_ik_result(arm_q, hand_q)
                if quest.data_dir is not None:
                    # Make sure the data_dir exists
                    os.makedirs(quest.data_dir, exist_ok=True)
                    if args.no_camera:
                        point_cloud = np.zeros((1000,3)) # dummy point cloud
                    if args.handedness == "right":
                        np.savez(f"{quest.data_dir}/right_data_{time.time()}.npz", right_wrist_pos=wrist_pos, right_wrist_orn=wrist_orn, 
                                                                                head_pos=head_pos, head_orn=head_orn.as_quat(),
                                                                                right_arm_q=arm_q, right_hand_q=action,raw_hand_q=hand_q,
                                                                                right_tip_poses=hand_tip_pose, point_cloud=point_cloud)
                    else:
                        np.savez(f"{quest.data_dir}/left_data_{time.time()}.npz", left_wrist_pos=wrist_pos, left_wrist_orn=wrist_orn, 
                                                                                head_pos=head_pos, head_orn=head_orn.as_quat(),
                                                                                left_arm_q=arm_q, left_hand_q=action, raw_hand_q=hand_q,
                                                                                left_tip_poses=hand_tip_pose, point_cloud=point_cloud)
        except socket.error as e:
            print(e)
            pass
        except KeyboardInterrupt:
            running = False
            quest.close()
            if not args.no_camera:
                camera.close()
            break
        else:
            packet_time = time.time()
            fps_counter += 1
            packet_counter += 1

            if (packet_time - start_time) > 1.0:
                print(f"received {fps_counter} packets in a second", end="\r")
                start_time += 1.0
                fps_counter = 0


