"""
generate_yolo_dataset_cleaned.py: Generates the YOLO format based on our custom dataset.
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
import os

import cv2
import imageio as iio
import numpy as np
import yaml


def calculate_2d_bbox(segmentation_map):
    segmentation_color = np.all(segmentation_map == (0, 0, 0), axis=-1)
    non_zero_indices = np.nonzero(segmentation_color)

    if non_zero_indices[0].size == 0 or non_zero_indices[1].size == 0:
        return None

    x_min, x_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    y_min, y_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_size = x_max - x_min
    y_size = y_max - y_min

    return x_center, y_center, x_size, y_size


def match_files(rgb_files, segmentation_files):
    matched_files = {}
    for rgb_file in rgb_files:
        rgb_scene_number = rgb_file.split('.')[0]
        for seg_file in segmentation_files:
            seg_scene_number = seg_file.split('.')[0]
            if rgb_scene_number == seg_scene_number:
                matched_files[rgb_file] = seg_file
                break
        else:
            print(f"No match found for {rgb_file}")
    if not matched_files:
        print("No matches found at all.")
    return matched_files


def get_class_for_frame(frame_number, classes):
    for class_info in classes:
        if class_info['start_frame'] <= frame_number <= class_info['end_frame']:
            return class_info['object_class'], class_info['class']
    return None, None


def handle_image(image_path, label_path, output_path, classes, visualize=True):
    img = cv2.imread(image_path)
    lbl = iio.imread(label_path)

    img_height, img_width = img.shape[:2]

    # Calculate bounding box
    bbox = calculate_2d_bbox(lbl)

    if bbox is not None:
        x_center, y_center, x_size, y_size = bbox

        # Normalize bounding box values
        x_center /= img_width
        y_center /= img_height
        x_size /= img_width
        y_size /= img_height

        # Get class
        frame_number = int(os.path.splitext(os.path.basename(image_path))[0])
        _, class_index = get_class_for_frame(frame_number, classes)

        with open(output_path, 'w') as f:
            f.write(f"{class_index} {x_center} {y_center} {x_size} {y_size}\n")

        if visualize:
            # Convert bounding box coordinates back to original image dimensions
            x_center *= img_width
            y_center *= img_height
            x_size *= img_width
            y_size *= img_height

            top_left = (int(x_center - x_size / 2), int(y_center - y_size / 2))
            bottom_right = (int(x_center + x_size / 2), int(y_center + y_size / 2))

            # Draw rectangle on image
            img_with_bbox = cv2.rectangle(img.copy(), top_left, bottom_right, (0, 255, 0), 2)

            # Display the image
            cv2.imshow('Image with BBox', img_with_bbox)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def generate_yolo_dataset(images_folder, labels_folder, json_file, output_folder):
    with open(json_file, 'r') as f:
        classes = json.load(f)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    rgb_files = [f for f in os.listdir(images_folder) if f.endswith('.bmp')]
    segmentation_files = [f for f in os.listdir(labels_folder) if
                          f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.jpeg')]

    matched_files = match_files(rgb_files, segmentation_files)

    for rgb_file, seg_file in matched_files.items():
        image_path = os.path.join(images_folder, rgb_file)
        label_path = os.path.join(labels_folder, seg_file)
        output_path = os.path.join(output_folder, os.path.splitext(rgb_file)[0] + '.txt')
        handle_image(image_path, label_path, output_path, classes)


def create_yolo_class_file(json_file, output_file, dataset_dir, train_dir, val_dir):
    with open(json_file, 'r') as f:
        classes = json.load(f)

    class_names = {class_info['class']: class_info['object_class'] for class_info in classes}
    class_names = dict(sorted(class_names.items()))  # Sort by class id

    config = {
        '# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]': None,
        'path': dataset_dir,
        'train': train_dir,
        'val': val_dir,
        'test': None,
        '# Classes': None,
        'names': class_names,
    }

    with open(output_file, 'w') as f:
        yaml.dump(config, f)


generate_yolo_dataset('', '', '', '')
create_yolo_class_file('', '', '', '', '')
