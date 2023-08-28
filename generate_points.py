"""
generate_points_cleaned.py: Generates the RGB point clouds based on the depth image and RGB image.
Copyright (C) 2023  Shromm Gaind

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import math
import os
import re

import cv2
import imageio as iio
import numpy as np
import open3d as o3d


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    o3d.visualization.draw_geometries_with_editing([pcd])


def read_camera_pose(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()

    match = re.search(
        r"Camera Location: X=([-]?\d+\.?\d*) Y=([-]?\d+\.?\d*) Z=([-]?\d+\.?\d*), Camera Rotation: P=([-]?\d+\.?\d*) Y=([-]?\d+\.?\d*) R=([-]?\d+\.?\d*)",
        log_data)

    if match:
        translation = np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
        rotation = euler_to_rotation_matrix(float(match.group(4)), float(match.group(5)), float(match.group(6)))
        return translation, rotation
    else:
        print('No camera pose found in log file.')
        return None


def euler_to_rotation_matrix(pitch, yaw, roll):
    # Convert degrees to radians
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)

    # Compute rotation matrix
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def depth_conversion(PointDepth, f):
    H, W = PointDepth.shape
    i_c = float(H) / 2 - 1
    j_c = float(W) / 2 - 1
    cols = np.linspace(0, W - 1, num=W)
    rows = np.linspace(0, H - 1, num=H).reshape(-1, 1)  # reshape to column vector for broadcasting
    DistanceFromCenter = np.sqrt((rows - i_c) ** 2 + (cols - j_c) ** 2)
    PlaneDepth = PointDepth / np.sqrt(1 + (DistanceFromCenter / f) ** 2)
    return PlaneDepth


def transform_point_cloud(pc, translation, rotation):
    # Add ones to the pc matrix
    ones = np.ones((pc.shape[0], 1))
    pc_homo = np.hstack((pc, ones))

    # Create the transformation matrix
    transform_matrix = np.zeros((4, 4))
    transform_matrix[0:3, 0:3] = rotation
    transform_matrix[0:3, 3] = translation
    transform_matrix[3, 3] = 1

    # Apply the transformation matrix
    transformed_pc_homo = np.dot(transform_matrix, pc_homo.T).T

    # Convert from homogeneous coordinates to 3D
    transformed_pc = transformed_pc_homo[:, 0:3] / transformed_pc_homo[:, 3].reshape(-1, 1)
    return transformed_pc


def create_point_cloud(folder_path, folder_path2, rgb_folder_path):
    depth_image_paths = {int(re.findall(r'\d+', file)[0]): os.path.join(folder_path, file)
                         for file in os.listdir(folder_path) if file.endswith(".png")}

    rgb_image_paths = {int(re.findall(r'\d+', file)[0]): os.path.join(rgb_folder_path, file)
                       for file in os.listdir(rgb_folder_path) if file.endswith(".bmp")}

    index_map = {key: (depth_path, rgb_image_paths[key])
                 for key, depth_path in depth_image_paths.items() if key in rgb_image_paths}

    for key, (depth_path, rgb_path) in index_map.items():
        # Read depth image as float32 and convert
        depth_image = iio.imread(depth_path)

        # Visualize depth image
        # plt.imshow(depth_image)
        # plt.show()

        # Read camera pose
        # translation, rotation = read_camera_pose(f'/home/eflinspy/Dataset/Pigonbelly90FOV/test/Text File.txt')

        # Correct depth calculation from RGB to depth values
        # Correct depth calculation from RGB to depth values and convert from mm to meters
        pixel_depth = (depth_image[:, :, 0] + depth_image[:, :, 1] * 256 + depth_image[:, :, 2] * 256 * 256)
        # Camera parameters (intrinsics)
        FX_DEPTH = 320  # Focal length along X-axis
        FY_DEPTH = 320  # Focal length along Y-axis
        CX_DEPTH = 640 / 2  # The x-coordinate of the principal point (usually the image width / 2)
        CY_DEPTH = 480 / 2  # The y-coordinate of the principal point (usually the image height / 2)

        # Convert from radial depth to Cartesian depth
        pixel_depth = depth_conversion(pixel_depth, FX_DEPTH)

        # Get indices where pixel_depth is nonzero
        i, j = np.where(pixel_depth != 0)

        # Proceed with point cloud calculation only if there are nonzero depth pixels
        if len(i) > 0 and len(j) > 0:
            depth_object = pixel_depth[i, j]

            # Compute the 3D coordinates using camera parameters
            x = ((j - CX_DEPTH) * depth_object / FX_DEPTH) / 1000
            y = ((i - CY_DEPTH) * depth_object / FY_DEPTH) / 1000
            z = depth_object / 1000
            # Check if the arrays are not empty

            # if len(x) > 0 and len(y) > 0 and len(z) > 0:
            #    print(f"The first point in the point cloud is: X={x[0]}, Y={y[0]}, Z={z[0]} meters")
            # else:
            #    print("No points in the point cloud.")

            # Transform points to camera coordinates
            points_camera = np.vstack((x, y, z)).T

            # Read camera pose
            # translation, rotation = read_camera_pose('/home/eflinspy/Dataset/training/test/logtest.txt')

            # Transform point cloud to world coordinates
            # points_world = transform_point_cloud(points_camera, translation, rotation)

            # Read the RGB image using OpenCV
            rgb_image = cv2.imread(rgb_path)

            # Convert the image from BGR to RGB format
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

            # Compute coordinates u and v (after filtering out points where z=0)
            non_zero_z_indices = np.where(z)
            x_nz, y_nz, z_nz = x[non_zero_z_indices], y[non_zero_z_indices], z[non_zero_z_indices]
            u = np.clip(np.round(FX_DEPTH * x_nz / z_nz + CX_DEPTH).astype(int), 0, rgb_image.shape[1] - 1)
            v = np.clip(np.round(FY_DEPTH * y_nz / z_nz + CY_DEPTH).astype(int), 0, rgb_image.shape[0] - 1)

            # Fetch RGB values based on u and v
            rgb_values = rgb_image[v, u, :].astype(np.float32) / 255.0

            # Create an array to hold the complete data
            points_with_color = np.zeros((points_camera.shape[0], 6), dtype=np.float32)
            points_with_color[:, :3] = points_camera
            points_with_color[non_zero_z_indices, 3:] = rgb_values

        # Save the point cloud with color to a binary file
        bin_filename = os.path.basename(depth_path).split('.')[0] + '.bin'
        points_with_color.astype('float32').tofile(os.path.join(folder_path2, bin_filename))

        # Convert the numpy array to an open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_camera)
        pcd.colors = o3d.utility.Vector3dVector(rgb_values)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

        """
        # Convert the numpy array to an open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_camera)
        pcd.colors = o3d.utility.Vector3dVector(rgb_values)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])

        # Create the bounding box
        centroid = np.array([ 0.03917045, -0.48140668, 3.54777893])
        dimensions = np.array([0.80742061, 0.87473894, 1.07651756])

        # Define the eight corners of the bounding box
        min_bound = centroid - dimensions / 2
        max_bound = centroid + dimensions / 2
        corners = np.array([[min_bound[0], min_bound[1], min_bound[2]],
                            [min_bound[0], min_bound[1], max_bound[2]],
                            [min_bound[0], max_bound[1], min_bound[2]],
                            [min_bound[0], max_bound[1], max_bound[2]],
                            [max_bound[0], min_bound[1], min_bound[2]],
                            [max_bound[0], min_bound[1], max_bound[2]],
                            [max_bound[0], max_bound[1], min_bound[2]],
                            [max_bound[0], max_bound[1], max_bound[2]]])

        # Define the edges of the bounding box
        edges = np.array([[0, 1], [0, 2], [0, 4],
                          [1, 3], [1, 5], [2, 3],
                          [2, 6], [3, 7], [4, 5],
                          [4, 6], [5, 7], [6, 7]])

        # Create the LineSet object representing the bounding box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        # Create a duplicate set of lines to achieve thicker lines
        thick_lines = np.vstack((edges, edges + 8))
        line_set.lines = o3d.utility.Vector2iVector(thick_lines)

        line_colors = np.tile([1.0, 0.0, 0.0], (len(edges) * 2, 1))
        line_set.colors = o3d.utility.Vector3dVector(line_colors)  # Set the color to red

        # Visualize the point cloud with the colored bounding box
        o3d.visualization.draw_geometries([pcd, line_set])

        # Add this line to allow point picking
        pick_points(pcd)
        """
        with open(os.path.join(folder_path2, bin_filename), 'rb') as f:
            num_points = os.path.getsize(os.path.join(folder_path2, bin_filename)) // (6 * 4)
            print('Number of points in {}: {}'.format(bin_filename, num_points))


# Define the paths to the depth and RGB image folders
folder_path = '' #depth image part
rgb_path = '' # rgb image path
folder_path2 = '' #pointscloud save path

# centroid = np.array([0.48068095, -0.62253954, 2.24203113])
# dimensions = np.array([1.95770259, 0.96200655, 0.67337449])
# yaw =  -1.320270357721661


create_point_cloud(folder_path, folder_path2, rgb_path)
