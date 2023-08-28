# Conference Paper Code Repository

This repository contains the code accompanying the conference paper. Below are descriptions and instructions for each Python script.

## Files

### [generate_points.py](./generate_points.py)
Generates the RGB point clouds based on the depth image and RGB image.

#### Paths to Modify
- `folder_path`: Path to the folder containing depth images.
- `rgb_path`: Path to the folder containing RGB images.
- `folder_path2`: Path to the folder where points will be stored.

#### Usage
```
python generate_points.py
```

### [sunrgbd_infos_old.py](./sunrgbd_infos_old.py)
Generates the 3D bounding box and puts the information to emulate the SUNRGBD dataset.

#### Paths to Modify
- `depth_folder`: Path to the folder containing depth files.
- `segmentation_folder`: Path to the folder containing segmentation files.
- `pc_fol`: Path to the folder containing points.

#### Usage
```
python sunrgbd_infos_old.py
```

### [generate_yolo_dataset.py](./generate_yolo_dataset.py)
Generates the YOLO format based on our custom dataset.

#### Paths to Modify
- `images_folder`: Path to the folder containing RGB images.
- `labels_folder`: Path to the folder containing label files.
- `output_folder`: Path to the folder where the output will be stored.
- `img_fol`: Path to the folder containing RGB images for testing.

#### Function Calls to Modify
- `def generate_yolo_dataset(images_folder, labels_folder, json_file, output_folder):`
- `def create_yolo_class_file(json_file, output_file, dataset_dir, train_dir, val_dir):`
- `create_yolo_class_file('/home/eflinspy/Dataset/newtestset/validation/classes.json', '/home/eflinspy/Dataset/newtestset/validation/classes.txt', '/home/eflinspy/Dataset/newtestset/', '/home/eflinspy/Dataset/newtestset/training/rgb', '/home/eflinspy/Dataset/newtestset/validation/rgb')`

#### Usage
```
python generate_yolo_dataset.py
```

## Installation

To install all required packages, run the following command:

```
pip install -r requirements.txt
```

## License

This project is licensed under the Affero General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Docker Setup

### TR3D Network

1. **Clone the GitHub repository:**
    ```bash
    git clone https://github.com/SamsungLabs/tr3d.git
    ```

2. **Build the Docker image:**
    ```bash
    docker build -t mmdetection3d -f docker/Dockerfile .
    ```

3. **Run the Docker container:**
    ```bash
    docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
    ```

#### Folder Structure for TR3D
Move the `.pkl` files to `mmdetection3d/data/sunrgbd` and then follow the TR3D readme files in order to train the network.

---

### Yolov8 Network

1. **Pull the Docker image from Docker Hub:**
    ```bash
    docker pull ultralytics/ultralytics
    ```

#### Folder Structure for Yolov8
Make sure that the `.txt` file and `.bmp` files are in the same folder structure.

---

