import os
import shutil
from pathlib import Path

# python


def safe_copy(src: Path, dst_dir: Path, overwrite: bool = False) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and not overwrite:
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            new_name = f"{stem}_{i}{suf}"
            new_dst = dst_dir / new_name
            if not new_dst.exists():
                shutil.copy2(src, new_dst)
                return new_dst
            i += 1
    else:
        shutil.copy2(src, dst)
        return dst


def copy_tif_and_json(input_dir: Path, output_dir: Path, overwrite: bool = False, preserve_structure: bool = False):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for fname in files:
            if fname.lower().endswith((".tif", ".tiff")):
                tif_path = root_path / fname
                json_path = tif_path.with_suffix(".json")
                # 仅当同名 json 存在时才拷贝
                if not json_path.exists():
                    print(f"Skip (no json): {tif_path}")
                    continue
                rel_dir = root_path.relative_to(input_dir) if preserve_structure else Path(".")
                dst_dir = output_dir / rel_dir
                try:
                    copied_tif = safe_copy(tif_path, dst_dir, overwrite=overwrite)
                    copied_json = safe_copy(json_path, dst_dir, overwrite=overwrite)
                    print(f"Copied: {tif_path} -> {copied_tif}; {json_path} -> {copied_json}")
                except Exception as e:
                    print(f"Failed: {tif_path} ({e})")


def collect_files_flat(input_dir: Path, output_dir: Path, overwrite: bool = False):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for fname in files:
            src = root_path / fname
            dst = output_dir / fname

            try:
                if dst.exists():
                    if overwrite:
                        shutil.copy2(src, dst)
                        print(f"Overwritten: {src} -> {dst}")
                    else:
                        stem = dst.stem
                        suffix = dst.suffix
                        i = 1
                        while True:
                            new_name = f"{stem}_{i}{suffix}"
                            new_dst = output_dir / new_name
                            if not new_dst.exists():
                                shutil.copy2(src, new_dst)
                                print(f"Copied (renamed): {src} -> {new_dst}")
                                break
                            i += 1
                else:
                    shutil.copy2(src, dst)
                    print(f"Copied: {src} -> {dst}")
            except Exception as e:
                print(f"Failed: {src} ({e})")


if __name__ == "__main__":
    input_path = r"D:\result_low_conf"
    output_path = r"D:\\needle_json_2026_0119_version"
    # collect_files_flat(input_path, output_path, overwrite=False)
    copy_tif_and_json(input_path, output_path)
