import matplotlib.pylab as plt
import numpy as np
import pandas as pd

# def deal_yolov7_result(data_path):
#     with open(data_path) as f:
#         data = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())))
#     return data


def _read_results_csv(path: str) -> pd.DataFrame:
    """Read a Ultralytics results.csv robustly and normalize headers."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # strip leading/trailing spaces from headers
    df.columns = df.columns.str.strip()
    return df


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first existing column name from candidates; raise if none found."""
    cols = set(df.columns)
    for name in candidates:
        if name in cols:
            return name
    # Try relaxed match without slashes or case
    lowered = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    raise KeyError(f"None of the candidate columns found in CSV. Candidates={candidates}, available={list(df.columns)}")


def _get_epoch_x(df: pd.DataFrame) -> np.ndarray:
    """Use 'epoch' column if present; otherwise range(len(df))."""
    if "epoch" in df.columns:
        return df["epoch"].to_numpy()
    return np.arange(len(df))


if __name__ == "__main__":
    # Input CSV paths
    # dinov3_result_2_csv = r'E:\Human_Projects\yolov11\runs\segment\train9\results.csv'
    yolov11_result_csv = (
        r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train6_origin_yolov11n\results.csv"
    )
    dinov3_result_csv = (
        r"E:\Human_Projects\yolov11\ultralytics-main\ultralytics\runs\segment\train7_my_experiment_6\results.csv"
    )

    # Read CSVs robustly
    yolov11_result_data = _read_results_csv(yolov11_result_csv)
    dinov3_result_data = _read_results_csv(dinov3_result_csv)
    # dinov3_result_2_data = _read_results_csv(dinov3_result_2_csv)

    # Resolve column names for all metrics
    metric_candidates = {
        "mAP50": ["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5", "mAP_0.5", "mAP50"],
        "precision": ["metrics/precision(B)", "metrics/precision", "precision"],
        "recall": ["metrics/recall(B)", "metrics/recall", "recall"],
        "mAP50-95": ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP_0.5:0.95", "mAP50-95"],
    }
    y11_cols = {k: _resolve_col(yolov11_result_data, v) for k, v in metric_candidates.items()}
    dino_cols = {k: _resolve_col(dinov3_result_data, v) for k, v in metric_candidates.items()}
    # dino2_cols = {k: _resolve_col(dinov3_result_2_data, v) for k, v in metric_candidates.items()}

    # X-axis from epoch column if available
    x_y11 = _get_epoch_x(yolov11_result_data)
    x_dino = _get_epoch_x(dinov3_result_data)
    # x_dino2 = _get_epoch_x(dinov3_result_2_data)

    # Plot 4 metrics in a row
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    metric_titles = ["mAP@0.5", "Precision", "Recall", "mAP@0.5:0.95"]
    metric_keys = ["mAP50", "precision", "recall", "mAP50-95"]
    for idx, (ax, title, key) in enumerate(zip(axs, metric_titles, metric_keys)):
        ax.plot(x_y11, pd.to_numeric(yolov11_result_data[y11_cols[key]], errors="coerce"), label="yolov11", linewidth=2)
        ax.plot(
            x_dino, pd.to_numeric(dinov3_result_data[dino_cols[key]], errors="coerce"), label="dino_yolo", linewidth=2
        )
        # ax.plot(x_dino2, pd.to_numeric(dinov3_result_2_data[dino2_cols[key]], errors='coerce'), label='dino_yolo_v2', linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig("metrics_curve_row.png")

    data_dict = {
        "yolov11": [0.672, 0.1 + 3.2 + 0.7, "+"],
        "dino_yolo": [0.74, 4.0, "*"],
        # 'dino_yolo_v2':[0.711, 4.5, 'x'],
        # 'dino_yolo':[0.772, 9.9, 'D'],
        # 'yolov10n':[0.727, 5.3, '_']
    }

    plt.figure(figsize=(10, 8))  # 调整图形大小
    for model_name in data_dict:
        print(data_dict[model_name][1], data_dict[model_name][0])
        plt.scatter(
            data_dict[model_name][1], data_dict[model_name][0], label=model_name, marker=data_dict[model_name][2], s=500
        )
    plt.xlabel("Inference Time(ms/img)", fontsize=14)  # 调整x轴标签字体大小
    plt.ylabel("mAP@0.5", fontsize=14)  # 调整y轴标签字体大小
    plt.legend(fontsize=20, loc=4)  # 调整图例字体大小
    plt.xticks(fontsize=12)  # 调整x轴刻度字体大小
    plt.yticks(fontsize=12)  # 调整y轴刻度字体大小
    plt.title("inferencetimevsmAP50", fontsize=20)
    plt.tight_layout()
    plt.savefig("inferencetimevsmAP50.png")
