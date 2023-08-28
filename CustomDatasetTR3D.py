"""
sunrgbd_infos_old_cleaned.py: Generates the 3D bounding box and puts the information to emulate the SUNRGBD dataset.
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
import json
import math
import os
import pickle
import re

import imageio as iio
import numpy as np
import open3d as o3d

with open('', 'r') as f:
    classes = json.load(f)


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


def calculate_2d_bbox(segmentation_map):
    segmentation_color = np.all(segmentation_map == (0, 0, 0), axis=-1)
    y_idx, x_idx = np.where(segmentation_color)

    if len(x_idx) == 0 or len(y_idx) == 0:
        return None

    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min, y_max = np.min(y_idx), np.max(y_idx)

    x_size = x_max - x_min
    y_size = y_max - y_min

    return x_min, y_min, x_size, y_size


def match_files(depth_files, segmentation_files):
    matched_files = {}
    for depth_file in depth_files:
        depth_scene_number = depth_file.split('.')[0]  # Split on '.' and get the first part
        for seg_file in segmentation_files:
            seg_scene_number = seg_file.split('.')[0]  # Split on '.' and get the first part
            if depth_scene_number == seg_scene_number:
                matched_files[depth_file] = seg_file
                break
        else:  # This will run if the for loop didn't break, i.e., if no match was found
            print(f"No match found for {depth_file}")
    if not matched_files:
        print("No matches found at all.")
    return matched_files


def draw_bounding_box(bbox, color=[1, 0, 0]):
    # Create line set
    line_set = o3d.geometry.LineSet()

    # The bounding box vertices
    vertices = np.asarray(bbox.get_box_points())

    # Lines connecting the vertices
    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5],
        [2, 3], [2, 6], [3, 7], [4, 5], [4, 6],
        [5, 7], [6, 7]
    ]

    # Create the line set from vertices and lines
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    return line_set


def create_oriented_bounding_box(pointcloud):
    # Create a PointCloud object from the input pointcloud data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    # Compute the axis-aligned bounding box
    aabb = pcd.get_axis_aligned_bounding_box()

    # Extract the center and dimensions of the AABB
    centroid = np.array(aabb.get_center())
    dimensions = np.array(aabb.get_extent())

    return centroid, dimensions


def visualize_oriented_bounding_box(centroid, rotation_matrix, dimensions, pointcloud):
    # Create an oriented bounding box with the extracted parameters
    obb = o3d.geometry.OrientedBoundingBox(center=centroid, R=rotation_matrix, extent=dimensions)

    # Create a LineSet for the bounding box with specified color
    bbox_lineset = draw_bounding_box(obb)  # Red color

    # Create a PointCloud object from the input pointcloud data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    # Visualize the pointcloud and oriented bounding box
    o3d.visualization.draw_geometries([pcd, bbox_lineset])


def get_yaw(rotation_matrix):
    # Assuming the Euler angles (roll, pitch, yaw)
    if rotation_matrix[0, 0] == 0 and rotation_matrix[1, 0] == 0:  # singularity
        theta = np.pi / 2 if rotation_matrix[0, 0] <= 0 else -np.pi / 2
    else:
        theta = np.arctan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
    return theta


def get_class_for_frame(frame_number, classes):
    for class_info in classes:
        if class_info['start_frame'] <= frame_number <= class_info['end_frame']:
            return class_info['object_class'], class_info['class']
    return None, None


depth_folder = ''
segmentation_folder = ''

depth_files = sorted(os.listdir(depth_folder))
segmentation_files = sorted(os.listdir(segmentation_folder))

label_pointclouds = {}

FX_DEPTH = 320
FY_DEPTH = 320
CX_DEPTH = 320
CY_DEPTH = 240
focal_length = FX_DEPTH

sorted_label_pointclouds = {}

matched_files = match_files(depth_files, segmentation_files)

info_list = []

# Continue with your existing code
for depth_file, seg_file in matched_files.items():
    match = re.search(r"(\d+)", depth_file)
    if match:
        depth_scene_number = match.group(1)
        label_pointclouds[depth_scene_number] = {}
        obj_id = 0

        # Read camera pose
        # translation, rotation = read_camera_pose(f'/home/eflinspy/Dataset/Pigonbelly90FOV/test/Text File.txt')

        depth_image = iio.v3.imread(os.path.join(depth_folder, depth_file)).astype(np.float32)
        segmentation_map = iio.v3.imread(os.path.join(segmentation_folder, seg_file))

        # Calculate the 2D bounding box
        bbox_2d = calculate_2d_bbox(segmentation_map)
        # If there is no object with RGB value (0, 0, 0), skip the current image
        if bbox_2d is None:
            continue
        if bbox_2d:
            print(f"2D bounding box for depth scene {depth_scene_number}: {bbox_2d}")

        pixel_depth_mm = (depth_image[:, :, 0] + depth_image[:, :, 1] * 256 + depth_image[:, :, 2] * 256 * 256)
        # Convert from radial depth to Cartesian depth
        pixel_depth = depth_conversion(pixel_depth_mm, FX_DEPTH)
        # Compute the 3D coordinates using camera parameters for entire image
        H, W = pixel_depth.shape
        i = np.arange(H)[:, None]
        j = np.arange(W)
        x = ((j - CX_DEPTH) * pixel_depth / FX_DEPTH) / 1000
        y = ((i - CY_DEPTH) * pixel_depth / FY_DEPTH) / 1000
        z = pixel_depth / 1000

        # Create point cloud from depth image
        points_camera = np.dstack((x, y, z))

        # Segmentation in camera coordinates
        segmentation_color = np.all(segmentation_map == (0, 0, 0), axis=-1)

        # Get segmented points
        segmented_points_camera = points_camera[segmentation_color]

        # Calculate the cutoff value as a percentage of the max value in each axis
        percentile_x_low, percentile_x_high = np.percentile(segmented_points_camera[:, 0], [0.3, 99.7])
        percentile_y_low, percentile_y_high = np.percentile(segmented_points_camera[:, 1], [0.3, 99.7])
        percentile_z_low, percentile_z_high = np.percentile(segmented_points_camera[:, 2], [0.3, 99.7])

        points_world_cutoff = segmented_points_camera[
            (segmented_points_camera[:, 0] > percentile_x_low) & (segmented_points_camera[:, 0] < percentile_x_high) &
            (segmented_points_camera[:, 1] > percentile_y_low) & (segmented_points_camera[:, 1] < percentile_y_high) &
            (segmented_points_camera[:, 2] > percentile_z_low) & (segmented_points_camera[:, 2] < percentile_z_high)]

        # Convert to Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world_cutoff)

        # Apply Statistical Outlier Removal
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=150, std_ratio=1.0)

        # Convert back to numpy array
        points_world_cutoff = np.asarray(pcd.points)

        # Store the 3D coordinates in the dictionary
        label_pointclouds[depth_scene_number][str(obj_id)] = points_world_cutoff
        obj_id += 1

# Here's the removed redundant calculation
calculated_centroids = {}
calculated_dimensions = {}
calculated_yaws = {}

for depth_scene_number, objects in label_pointclouds.items():
    for obj_id, pointcloud in objects.items():

        if pointcloud.size > 0:
            centroid, dimensions = create_oriented_bounding_box(pointcloud)

            # Compute yaw angle: the angle in the ground plane (XZ plane) between the x-axis and the projection of the first principal component onto the ground plane.
            # We use atan2 to get the correct quadrant.
            # theta = get_yaw(rotation_matrix)
            # print(f"Yaw for depth scene {depth_scene_number}: {theta} radians")

            # calculated_yaws[depth_scene_number] = rotation_matrix
            calculated_centroids[depth_scene_number] = centroid
            calculated_dimensions[depth_scene_number] = dimensions

            print(f"Centroid for depth scene {depth_scene_number}: {centroid}")
            print(f"Dimensions for depth scene {depth_scene_number}: {dimensions}")
            # print(f"Yaw for depth scene {depth_scene_number}: {rotation_matrix} radians")
            # visualize_oriented_bounding_box(centroid, rotation_matrix, dimensions, pointcloud)
        else:
            print(f"Pointcloud for depth scene {depth_scene_number}, object {obj_id} is empty.")

pc_fol = ""
img_fol = ""

info_list = []

# Create a mapping between image index and point cloud index
idx_mapping = {}
for pc_file in os.listdir(pc_fol):
    match = re.search(r'(\d+)\.bin', pc_file)
    if match:
        pointcloud_id = int(match.group(1))
        for img_file in os.listdir(img_fol):
            match = re.search(r'(\d+)\.bmp', img_file)
            if match:
                image_id = int(match.group(1))
                if pointcloud_id == image_id:
                    idx_mapping[image_id] = pointcloud_id

for depth_scene_number, objects in label_pointclouds.items():
    for obj_id, pointcloud in objects.items():
        object_class, class_number = get_class_for_frame(int(depth_scene_number), classes)
        print(f"Class for depth scene {depth_scene_number}: {object_class}")
        print(f"Class number: {class_number}")
        full_info = np.array([np.concatenate((calculated_centroids[depth_scene_number],
                                              calculated_dimensions[depth_scene_number],
                                              [0]))])
        if int(depth_scene_number) in idx_mapping:
            info = {}
            info['point_cloud'] = {'num_features': 1, 'lidar_idx': int(depth_scene_number)}
            info['pts_path'] = os.path.join(pc_fol, f"{depth_scene_number}.bin")
            info['image'] = {'image_idx': int(depth_scene_number), 'image_shape': np.array(depth_image.shape),
                             'image_path': os.path.join(img_fol, f"{depth_scene_number}.bmp")}
            FX_DEPTH = 320
            FY_DEPTH = 320
            CX_DEPTH = 320
            CY_DEPTH = 240
            K = np.array([[FX_DEPTH, 0, CX_DEPTH],
                          [0, FY_DEPTH, CY_DEPTH],
                          [0, 0, 1]])
            Rt = np.identity(4)
            info['calib'] = {'K': K, 'Rt': Rt}
            annotations = {
                'gt_num': 1,
                'name': np.array([object_class]),
                'bbox': np.array([bbox_2d]),
                'location': np.vstack((calculated_centroids[depth_scene_number])),
                'dimensions': np.vstack((calculated_dimensions[depth_scene_number])),
                'rotation_y': np.array(0),
                'index': np.array([obj_id]),
                'class': np.array([class_number]),
                'gt_boxes_upright_depth': full_info,
            }
            info['annos'] = annotations
            info_list.append(info)

# Save the info_list to a pickle file
with open('sunrgbd_infos_val.pkl', 'wb') as handle:
    pickle.dump(info_list, handle, protocol=4)
