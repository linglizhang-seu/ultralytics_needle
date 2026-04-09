import json
import random
import shutil
from pathlib import Path


def flatten_directory():
    """将源目录下（包含所有子目录）的所有图片文件拷贝到同一个目标目录中。 解决目录层级过深或图片分散的问题。.
    """
    # ================= 配置区域 =================
    # 1. 源目录：包含大量子文件夹和图片的根目录
    source_root = r"D:\snapshot_result"

    # 2. 目标目录：所有图片将平铺到这里
    target_dir = r"D:\snapshot_result_images"
    # ===========================================

    source_path = Path(source_root)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 常见图片格式
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    count = 0
    print(f"开始扫描目录: {source_path}")

    # rglob('*') 会递归查找所有文件
    for file_path in source_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                # 构造目标文件路径
                # 注意：如果有重名文件，shutil.copy2 会直接覆盖，或者你可以加个前缀
                # 简单的防重名策略：加上父文件夹名作为前缀
                new_filename = f"{file_path.parent.name}_{file_path.name}"
                dest_path = target_path / new_filename

                shutil.copy2(file_path, dest_path)
                count += 1

                if count % 100 == 0:
                    print(f"已拷贝 {count} 张图片...")
            except Exception as e:
                print(f"拷贝失败: {file_path} -> {e}")

    print(f"\n操作完成！共拷贝 {count} 张图片到: {target_dir}")


def match_and_copy_files():
    """遍历A文件夹中的图像，在B文件夹中找到同名（主文件名相同）的文件，并拷贝到C文件夹。 常用于：根据筛选后的图片(A)，从总标签库(B)中提取对应的标签到(C)。.
    """
    # ================= 配置区域 =================
    # 1. 参照目录 (A)：一般是筛选过的图片目录
    # ref_dir_A = r"E:\Data\A_Selected_Images"
    ref_dir_A = r"D:\needle_negative2"

    # 2. 搜索目录 (B)：包含目标文件（如json, txt, 或原图）的目录
    # search_dir_B = r"E:\Data\B_All_Labels"
    search_dir_B = r"D:\needle0121_images"

    # 3. 输出目录 (C)
    # output_dir_C = r"E:\Data\C_Selected_Labels"
    output_dir_C = r"D:\needle0121_negative_images"
    # ===========================================

    ref_path = Path(ref_dir_A)
    search_path = Path(search_dir_B)
    output_path = Path(output_dir_C)

    # 支持的参照文件格式
    extensions_A = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if not ref_path.exists():
        print(f"错误：参照目录不存在 {ref_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"1. 正在扫描参照目录 A: {ref_path}")
    # 获取所有符合图片扩展名的文件的主文件名 (stem)
    ref_stems = set()
    for f in ref_path.iterdir():
        if f.is_file() and f.suffix.lower() in extensions_A:
            ref_stems.add(f.stem)

    print(f"   -> 找到了 {len(ref_stems)} 个唯一基准文件名。")

    if not ref_stems:
        print("警告：在 A 文件夹没找到图片，请检查路径。")
        return

    # 2. 遍历 B，查找匹配的文件并拷贝
    print(f"2. 正在扫描搜索目录 B: {search_path}")
    count = 0

    # 遍历 B 中所有文件
    # 这里我们遍历B中的所有文件，只要stem在A中就把B的文件拷走
    # 这样可以同时处理 .json 和 .txt
    if search_path.exists():
        for f in search_path.iterdir():
            if f.is_file():
                # 核心逻辑：如果 B 中文件的主文件名在 A 的列表中
                if f.stem in ref_stems:
                    try:
                        shutil.copy2(f, output_path / f.name)
                        count += 1
                        if count % 100 == 0:
                            print(f"   已拷贝 {count} 个文件...")
                    except Exception as e:
                        print(f"   拷贝失败: {f.name} -> {e}")
    else:
        print(f"错误：搜索目录 B 不存在 {search_path}")

    print("\n操作完成！")
    print(f"共从 B 拷贝了 {count} 个同名文件到 C: {output_dir_C}")


def copy_paired_images_and_json():
    """扫描源文件夹中的图片，检查是否存在同名的 JSON 文件。 如果存在 (img + json) 成对出现，则将两者都拷贝到目标文件夹。 用于筛选已标注的数据。.
    """
    # ================= 配置区域 =================
    # 1. 源目录：包含图片和 JSON 的混合文件夹
    source_dir = r"D:\snapshot"

    # 2. 目标目录：存放成对数据的文件夹
    output_dir = r"E:\Data\batches_finish"
    # ===========================================

    src_path = Path(source_dir)
    dst_path = Path(output_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if not src_path.exists():
        print(f"源目录不存在: {src_path}")
        return

    print(f"扫描目录: {src_path}")
    count = 0

    # 遍历源目录下的所有文件
    for f in src_path.iterdir():
        # 判断是否是图片
        if f.is_file() and f.suffix.lower() in img_extensions:
            # 构造对应的 JSON 文件路径
            json_file = f.with_suffix(".json")

            # 核心检查：JSON 是否存在
            if json_file.exists():
                try:
                    # 拷贝图片
                    shutil.copy2(f, dst_path / f.name)
                    # 拷贝 JSON
                    shutil.copy2(json_file, dst_path / json_file.name)

                    count += 1
                    if count % 100 == 0:
                        print(f"已处理 {count} 对文件...")
                except Exception as e:
                    print(f"拷贝失败 {f.name}: {e}")

    print(f"\n完成！共拷贝了 {count} 对 (图片+JSON) 到: {output_dir}")


def create_negative_samples_dataset():
    # ================= 配置区域 =================
    # 1. 原始纯背景图像所在的文件夹路径
    source_images_dir = r"E:\Data\yinxingduizhao"

    # 2. 输出数据集的根目录
    output_base_dir = r"E:\Data\negative_samples"

    # 3. 数据集划分比例 (需要和为 1.0)
    train_ratio = 0.9
    val_ratio = 0.1
    # ===========================================

    source_path = Path(source_images_dir)
    output_path = Path(output_base_dir)

    # 支持的图像扩展名
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    # 获取所有图像文件
    if not source_path.exists():
        print(f"错误: 源目录不存在 {source_path}")
        return

    images = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]

    if not images:
        print("未找到图像文件")
        return

    print(f"找到 {len(images)} 张图像，准备处理...")

    # 打乱顺序
    random.shuffle(images)

    # 计算数量
    total = len(images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    total - train_count - val_count

    # 划分列表
    splits = {
        "train": images[:train_count],
        "val": images[train_count : train_count + val_count],
        "test": images[train_count + val_count :],
    }

    print(f"划分情况: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # 创建目录结构并复制文件
    for split_name, split_images in splits.items():
        # 目标图片目录: datasets/images/train
        img_dest_dir = output_path / "images" / split_name
        # 目标标签目录: datasets/labels/train
        label_dest_dir = output_path / "labels" / split_name

        img_dest_dir.mkdir(parents=True, exist_ok=True)
        label_dest_dir.mkdir(parents=True, exist_ok=True)

        for img_file in split_images:
            # 1. 复制图像
            shutil.copy2(img_file, img_dest_dir / img_file.name)

            # 2. 生成同名空 txt 文件
            label_name = img_file.stem + ".txt"
            label_file = label_dest_dir / label_name

            # 创建空文件
            with open(label_file, "w"):
                pass  # 空文件代表该图无目标（阴性样本）

    print("\n处理完成！")
    print(f"数据集已生成至: {output_base_dir}")
    print("文件夹结构为 YOLO 标准格式:")
    print(f"  {output_base_dir}/images/train")
    print(f"  {output_base_dir}/labels/train")
    print("  ...")


def merge_needle_jsons():
    """合并两批 needle 相关的标注数据到新的文件夹： 1. tips_dir: 包含 'needle' 标签（实际为针尖），需要重命名为 'needle_tip' 2. total_dir: 包含 'needle'
    标签（全针），保持不变 将合并后的 JSON 和对应图片保存到 output_dir。.
    """
    # ================= 配置区域 =================
    # 1. 针尖数据目录 (Label=needle -> 改为 needle_tip)
    tips_dir = r"E:\Data\batch_finish_2\356_needle_tips"

    # 2. 全针数据目录 (Label=needle -> 保持 needle)
    total_dir = r"E:\Data\batch_finish_2\356_needle_total"

    # 3. 输出目录
    output_dir = r"E:\Data\batch_finish_2\356_merged"
    # ===========================================

    path_tips = Path(tips_dir)
    path_total = Path(total_dir)
    path_out = Path(output_dir)

    path_out.mkdir(parents=True, exist_ok=True)

    # 收集所有涉及的基准文件名 (stem)
    all_stems = set()

    # 扫描两个文件夹中的 JSON 文件
    if path_tips.exists():
        for p in path_tips.glob("*.json"):
            all_stems.add(p.stem)
    if path_total.exists():
        for p in path_total.glob("*.json"):
            all_stems.add(p.stem)

    print(f"共找到 {len(all_stems)} 个文件任务，开始合并...")

    count = 0
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    for stem in all_stems:
        # 定义输入输出路径
        json_tips_path = path_tips / f"{stem}.json"
        json_total_path = path_total / f"{stem}.json"

        base_data = None
        merged_shapes = []

        # 1. 处理针尖数据 (Tips)
        if json_tips_path.exists():
            try:
                with open(json_tips_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if base_data is None:
                        base_data = data

                    # 修改 label 并添加到列表
                    for shape in data.get("shapes", []):
                        if shape.get("label") == "needle":
                            shape["label"] = "needle_tip"
                        merged_shapes.append(shape)
            except Exception as e:
                print(f"读取 Tips JSON 失败 {stem}: {e}")

        # 2. 处理全针数据 (Total)
        if json_total_path.exists():
            try:
                with open(json_total_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if base_data is None:
                        base_data = data

                    # 保持 label 并添加到列表
                    for shape in data.get("shapes", []):
                        # 这里的 label 应该是 needle，保持不变
                        merged_shapes.append(shape)
            except Exception as e:
                print(f"读取 Total JSON 失败 {stem}: {e}")

        # 3. 保存合并结果
        if base_data:
            base_data["shapes"] = merged_shapes

            # 写入新的 JSON
            out_json = path_out / f"{stem}.json"
            try:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(base_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"写入 JSON 失败 {stem}: {e}")
                continue

            # 4. 拷贝对应的图片文件
            # 尝试从 tips 或 total 文件夹找到图片
            image_copied = False
            # 先找 tips
            for ext in img_extensions:
                img_src = path_tips / f"{stem}{ext}"
                if img_src.exists():
                    shutil.copy2(img_src, path_out / img_src.name)
                    image_copied = True
                    break

            # 如果没找到，找 total
            if not image_copied:
                for ext in img_extensions:
                    img_src = path_total / f"{stem}{ext}"
                    if img_src.exists():
                        shutil.copy2(img_src, path_out / img_src.name)
                        image_copied = True
                        break

            if image_copied:
                count += 1
                if count % 50 == 0:
                    print(f"已处理 {count} 个文件...")
            else:
                print(f"警告: 未找到图片文件 {stem}")

    print(f"\n合并完成！共生成 {count} 对文件在: {output_dir}")


if __name__ == "__main__":
    # merge_needle_jsons()
    create_negative_samples_dataset()
    # flatten_directory()
    # match_and_copy_files()
    # copy_paired_images_and_json()
