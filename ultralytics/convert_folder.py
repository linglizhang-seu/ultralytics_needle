import json
import random
try:
    import yaml
except ImportError:
    yaml = None
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(114514)

# Supported image formats for YOLO
image_formats = ["jpg", "jpeg", "png", "bmp", "webp", "tif", "dng", "mpo", "pfm"]

# 要跳过的标签（不写入 YOLO，也视为“空”）
SKIP_LABELS = {"contour", "needle_body"}

def normalize_label(label: str) -> str:
    # 目前不做合并，直接返回原标签
    return label

def copy_labled_img(json_path: Path, target_folder: Path, task: str):
    # Iterate through supported image formats and copy image files
    for fmt in image_formats:
        image_path = json_path.with_suffix("." + fmt)
        if image_path.exists():
            # Construct target path in the target folder
            target_path = target_folder / "images" / task / image_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(image_path, target_path)


def json_to_yolo(json_path: Path, sorted_keys: list):
    with open(json_path, "r", encoding="utf-8") as f:
        labelme_data = json.load(f)

    width = labelme_data.get("imageWidth", 1)
    height = labelme_data.get("imageHeight", 1)
    yolo_lines = []

    for shape in labelme_data.get("shapes", []):
        raw_label = shape.get("label", "")
        # 跳过指定标签，不写入 yolo
        if raw_label in SKIP_LABELS:
            continue

        label = normalize_label(raw_label)
        if label not in sorted_keys:
            # 如果出现未在名字列表中的标签，跳过
            continue

        points = shape.get("points", [])
        class_idx = sorted_keys.index(label)
        txt_string = "{} ".format(class_idx)

        for x, y in points:
            # 防止除以零
            x = float(x) / max(1.0, float(width))
            y = float(y) / max(1.0, float(height))
            txt_string += "{} {} ".format(x, y)

        yolo_lines.append(txt_string.strip() + "\n")

    return yolo_lines

def create_directory_if_not_exists(directory_path):
    directory_path.mkdir(parents=True, exist_ok=True)

# Create a YAML file for YOLO training
def create_yaml(output_folder: Path, sorted_keys: list):
    train_img_path = Path("images") / "train"
    val_img_path = Path("images") / "val"
    test_img_path = Path("images") / "test"
    train_label_path = Path("labels") / "train"
    val_label_path = Path("labels") / "val"
    test_label_path = Path("labels") / "test"

    # Create necessary directories
    for path in [train_img_path, val_img_path, test_img_path, train_label_path, val_label_path, test_label_path]:
        create_directory_if_not_exists(output_folder / path)

    names_dict = {idx: name for idx, name in enumerate(sorted_keys)}
    yaml_dict = {
        "path": output_folder.as_posix(),
        "train": train_img_path.as_posix(),
        "val": val_img_path.as_posix(),
        "test": test_img_path.as_posix(),
        "names": names_dict,
    }

    yaml_file_path = output_folder / "yolo.yaml"
    with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
        if yaml is not None:
            yaml.dump(yaml_dict, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        else:
            # 简单手动写入 YAML（当未安装 PyYAML 时）
            yaml_file.write(f"path: {yaml_dict['path']}\n")
            yaml_file.write(f"train: {yaml_dict['train']}\n")
            yaml_file.write(f"val: {yaml_dict['val']}\n")
            yaml_file.write(f"test: {yaml_dict['test']}\n")
            yaml_file.write("names:\n")
            for idx, name in names_dict.items():
                yaml_file.write(f"  {idx}: {name}\n")

def get_labels_and_json_path(input_folder: Path):
    json_file_paths = list(input_folder.rglob("*.json"))
    label_counts = defaultdict(int)

    for json_file_path in json_file_paths:
        with open(json_file_path, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)
        for shape in labelme_data.get("shapes", []):
            raw_label = shape.get("label", "")
            # 跳过不需要计数的标签
            if raw_label in SKIP_LABELS:
                continue
            label = normalize_label(raw_label)
            label_counts[label] += 1

    # 按出现次数降序排序并返回标签名称列表
    sorted_keys = sorted(label_counts, key=lambda k: label_counts[k], reverse=True)
    return sorted_keys, json_file_paths


def delete_empty_labels(json_file_paths: list, output_folder: Path):
    empty_label_files = []

    for json_file_path in json_file_paths:
        with open(json_file_path, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)

        # 只保留非跳过标签的 shapes
        remaining_shapes = [s for s in labelme_data.get("shapes", []) if s.get("label", "") not in SKIP_LABELS]
        if not remaining_shapes:  # 当排除跳过标签后没有任何标注时，视为 empty
            empty_label_files.append(json_file_path)

            # Copy image to 'empty' folder
            for fmt in image_formats:
                image_path = json_file_path.with_suffix("." + fmt)
                if image_path.exists():
                    image_copy_path = output_folder / "images" / "empty" / image_path.name
                    create_directory_if_not_exists(image_copy_path.parent)
                    shutil.copy(image_path, image_copy_path)
                    image_path.unlink()  # Delete the original image file

            json_file_path.unlink()  # Delete the original JSON file

    # Remove deleted files from the file list
    json_file_paths[:] = [path for path in json_file_paths if path not in empty_label_files]

    if empty_label_files:
        print("The following files with effectively empty labels have been deleted:")
        for file in empty_label_files:
            print(file)


def labelme_to_yolo(json_file_paths: list, output_folder: Path, sorted_keys: list):
    # Shuffle the list
    random.shuffle(json_file_paths)

    # Split the data
    total_files = len(json_file_paths)
    train_split_point = int(0.7 * total_files)
    val_split_point = train_split_point + int(0.2 * total_files)

    train_set = json_file_paths[:train_split_point]
    val_set = json_file_paths[train_split_point:val_split_point]
    test_set = json_file_paths[val_split_point:]

    # Process train set
    for json_file_path in tqdm(train_set, desc="Processing Train Set"):
        if not json_file_path.exists():  # Skip non-existent files
            print(f"Warning: File {json_file_path} does not exist. Skipping.")
            continue

        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "train" / txt_name)
        create_directory_if_not_exists(output_json_path.parent)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="train")

    # Process validation set
    for json_file_path in tqdm(val_set, desc="Processing Validation Set"):
        if not json_file_path.exists():
            print(f"Warning: File {json_file_path} does not exist. Skipping.")
            continue

        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "val" / txt_name)
        create_directory_if_not_exists(output_json_path.parent)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="val")

    # Process test set
    for json_file_path in tqdm(test_set, desc="Processing Test Set"):
        if not json_file_path.exists():
            print(f"Warning: File {json_file_path} does not exist. Skipping.")
            continue

        txt_name = json_file_path.with_suffix(".txt").name
        yolo_lines = json_to_yolo(json_file_path, sorted_keys)
        output_json_path = Path(output_folder / "labels" / "test" / txt_name)
        create_directory_if_not_exists(output_json_path.parent)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)
        copy_labled_img(json_file_path, output_folder, task="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="labelme2yolo")
    parser.add_argument("input_folder", help="输入LabelMe格式文件的文件夹")
    parser.add_argument("output_folder", help="输出YOLO格式文件的文件夹")

    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    sorted_keys, json_file_paths = get_labels_and_json_path(input_folder)
    # 确保有非跳过标签存在
    if not sorted_keys:
        print("未找到任何有效标签（非 contour / needle_tip / needle_body）。")
    else:
        delete_empty_labels(json_file_paths, output_folder)
        create_yaml(output_folder, sorted_keys)
        labelme_to_yolo(json_file_paths, output_folder, sorted_keys)