#!/usr/bin/env python3
"""
Basic ZED Wrapper

Author: @Zi-ang-Cao
Date: Dec 2024

Usage:
    python zed_module.py --visualize

How to setup the ZED SDK:
    - Download the ZED SDK from the Stereolabs website: https://www.stereolabs.com/developers/
    - Install the ZED SDK according to the instructions provided on the website
    - Make sure the ZED SDK Python API is installed via https://www.stereolabs.com/docs/app-development/python/install
    - Based on the result of `python3 get_python_api.py`, install the python sdk to new environment
        + E.g.: `python -m pip install --ignore-installed /usr/local/zed/pyzed-4.2-cp38-cp38-linux_x86_64.whl`
        + E.g.: `python -m pip install --ignore-installed /usr/local/zed/pyzed-4.1-cp310-cp310-linux_x86_64.whl`
        + E.g.: `python -m pip install --ignore-installed /usr/local/zed/pyzed-4.1-cp311-cp311-linux_x86_64.whl`
"""

import pyzed.sl as sl
import numpy as np
import open3d as o3d
import signal
import sys
import time
import click
from plyfile import PlyData
import cv2

def crop_pcd(pcd, min_bound, max_bound):
    # Crop the point cloud based on given minimum and maximum bounds
    pcd_pos = pcd[:,:3]
    pcd_color = pcd[:,3:]
    mask = np.logical_and(np.all(pcd_pos > min_bound, axis=1), np.all(pcd_pos < max_bound, axis=1))
    pcd_pos = pcd_pos[mask]
    pcd_color = pcd_color[mask]
    return np.hstack([pcd_pos, pcd_color])

def adjust_cam_settings(cam):
    # Adjust the camera settings to predefined values
    # TODO: make sure autofocus is turned off.
    # github.com/stereolabs/zed-sdk/blob/master/camera%20control/python/camera_control.py
    cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 5)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 8)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 4)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 12)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 50)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 5600)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)
    cam.set_camera_settings(sl.VIDEO_SETTINGS.LED_STATUS, 1)

# Depth camera related functions and constants
class ZEDCameraModule:
    def __init__(self, is_decimate=False, visualize=False, manual_visualize=False, verbose=True):
        # Initialize ZED camera parameters and settings
        self.SAVE_PLY = False  # Flag to save point cloud as PLY file
        self.verbose = verbose
        if self.verbose or visualize:
            self.compute_freq = True
        else:
            self.compute_freq = False

        # Set camera initialization parameters
        init = sl.InitParameters(
            depth_mode=sl.DEPTH_MODE.NEURAL,
            camera_fps=30,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            )
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")

        # Open the ZED camera
        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        res = sl.Resolution()
        res.width = 1280
        res.height = 720
        adjust_cam_settings(zed)  # Adjust camera settings
        camera_model = zed.get_camera_information().camera_model
        
        # Initialize point cloud and camera properties
        self.point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.res = res
        self.zed = zed
        self.is_decimate = is_decimate
        self.visualize = visualize
        self.manual_visualize = manual_visualize
        self.w, self.h = res.width, res.height
        self.first_run = True
        self.last_time = time.time()
        self.rgb_window_name = "ZED RGB Image"
        if self.visualize or self.manual_visualize:
            cv2.namedWindow(self.rgb_window_name, cv2.WINDOW_NORMAL)

        # Set up Open3D visualization if needed
        if self.visualize or self.manual_visualize:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="ZED Point Cloud Visualization")
            self.o3d_pcd = o3d.geometry.PointCloud()
            coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.vis.add_geometry(self.o3d_pcd)
            self.vis.add_geometry(coordframe)

    def load_world(self, delta_pos, delta_orn):
        # Load transformation information for point cloud to world transformation
        self.delta_pos = delta_pos
        self.delta_orn = delta_orn

    def to_world(self, pcd, head_pos, head_orn):
        # Transform point cloud to world coordinates
        pcd_pos = pcd[:, :3]
        pcd_color = pcd[:, 3:]
        pcd_world = head_orn.apply(self.delta_orn.apply(pcd_pos) + self.delta_pos) + head_pos
        return np.hstack([pcd_world, pcd_color])

    def receive(self):
        if self.compute_freq:
            # Receive point cloud data from the ZED camera
            current_time = time.time()
            delta_time = current_time - self.last_time
            self.last_time = current_time
            update_frequency = 1.0 / delta_time if delta_time > 0 else 0.0

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve the point cloud from the camera
            point_cloud = self.point_cloud
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.res)
            if self.SAVE_PLY:
                # Save the point cloud as a PLY file
                point_cloud.write("point_cloud.ply")

                # Read the saved PLY file
                plydata = PlyData.read("point_cloud.ply")
                vertex = plydata['vertex']
                verts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
                colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
            else:
                # Retrieve XYZ and RGB data from the point cloud
                xyz = point_cloud.get_data()[:, :, 0:3]
                rgba = np.ravel(point_cloud.get_data()[:, :, 3]).view('uint8').reshape((self.res.height, self.res.width, 4))
                rgb = rgba[:,:,:3]
                verts = xyz.reshape(-1, 3)
                colors = rgb.reshape(-1, 3) / 255.0

            # Show the RGB image in a separate window
            if self.visualize or self.manual_visualize:
                rgb_image = (colors.reshape(self.res.height, self.res.width, 3) * 255).astype(np.uint8)
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(self.rgb_window_name, bgr_image)
                cv2.waitKey(1)
            
            # Filter out invalid points (NaN or infinite values)
            valid_mask = np.isfinite(verts).all(axis=1)
            verts = verts[valid_mask]
            colors = colors[valid_mask]

            # Ensure verts and colors have the same length
            min_length = min(len(verts), len(colors))
            verts = verts[:min_length]
            colors = colors[:min_length]

            # Decimate the point cloud for visualization purposes
            downsampled_verts = verts[::10]
            downsampled_colors = colors[::10]
            if self.visualize:
                # Further decimate points for visualization
                vis_verts = downsampled_verts[::5]
                vis_colors = downsampled_colors[::5]
                if self.first_run:
                    # Initialize the Open3D point cloud for the first run
                    self.o3d_pcd = o3d.geometry.PointCloud()
                    self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
                    self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                    coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                    self.vis = o3d.visualization.Visualizer()
                    if self.compute_freq:
                        self.vis.create_window(window_name=f"ZED Point Cloud Visualization - Points: {len(vis_verts)} - Update Frequency: {update_frequency:.2f} Hz")
                    self.vis.add_geometry(self.o3d_pcd)
                    self.vis.add_geometry(coordframe)
                    self.first_run = False
                else:
                    # Update the Open3D point cloud with new data
                    self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
                    self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
                    self.vis.update_geometry(self.o3d_pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    self.vis.get_view_control().set_lookat([0, 0, 0])
                    self.vis.get_render_option().background_color = np.array([1, 1, 1])
            # else:
            #     if self.compute_freq:
            #         print(f"Received point cloud data - Points: {len(downsampled_verts)} - Update Frequency: {update_frequency:.2f} Hz")
            return np.hstack([downsampled_verts, downsampled_colors])

    def visualize_pcd(self, pcd):
        # Visualize the given point cloud using Open3D
        vis_verts = pcd.copy()[:,:3]
        vis_colors = pcd.copy()[:,3:]
        if self.first_run:
            # Initialize the Open3D point cloud for the first run
            self.o3d_pcd = o3d.geometry.PointCloud()
            self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
            self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            coordframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
            self.vis.add_geometry(self.o3d_pcd)
            self.vis.add_geometry(coordframe)
            self.first_run = False
        else:
            # Update the Open3D point cloud with new data
            self.o3d_pcd.points = o3d.utility.Vector3dVector(vis_verts)
            self.o3d_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
            self.vis.update_geometry(self.o3d_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        # Close the ZED camera and destroy the Open3D visualization window
        self.zed.close()
        if self.visualize:
            self.vis.destroy_window()
        if self.visualize or self.manual_visualize:
            cv2.destroyWindow(self.rgb_window_name)

def signal_handler(sig, frame):
    # Handle interrupt signal to safely close the camera
    print("\nInterrupted! Closing the camera...")
    camera.close()
    sys.exit(0)

@click.command()
@click.option('--is_decimate', is_flag=True, default=False, help='Enable point cloud decimation.')
@click.option('--visualize', is_flag=True, default=False, help='Enable visualization of the point cloud.')
@click.option('--manual_visualize', is_flag=True, default=False, help='Enable manual visualization mode.')
@click.option('--verbose', is_flag=True, default=False, help='Enable verbose output.')
def main(is_decimate, visualize, manual_visualize, verbose):
    # Initialize the ZED camera module and set up signal handling
    global camera
    camera = ZEDCameraModule(is_decimate=is_decimate, visualize=visualize, manual_visualize=manual_visualize, verbose=verbose)
    signal.signal(signal.SIGINT, signal_handler)
    while True:
        # Continuously receive point cloud data
        camera.receive()

if __name__ == "__main__":
    main()
