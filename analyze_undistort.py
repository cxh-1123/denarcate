"""
基于标定 npz（默认 calibration_result_v1.npz）分析去畸变几何：输出多张曲线/柱状图 PNG。
依赖：numpy, opencv-python, matplotlib
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Tuple

import cv2
import numpy as np

# 中文标题（Windows 常见字体）
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _corners_xy(corners: np.ndarray) -> np.ndarray:
    return corners.reshape(-1, 2).astype(np.float64)


def _line_fit_rms_max(pts: np.ndarray) -> Tuple[float, float]:
    if pts.shape[0] < 2:
        return 0.0, 0.0
    line = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    lv = line.reshape(-1)
    vx, vy, x0, y0 = float(lv[0]), float(lv[1]), float(lv[2]), float(lv[3])
    vlen = np.hypot(vx, vy)
    if vlen < 1e-12:
        return 0.0, 0.0
    d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx) / vlen
    return float(np.sqrt(np.mean(d * d))), float(np.max(d))


def _chessboard_straightness(corners: np.ndarray, pattern_size: Tuple[int, int]) -> Dict[str, float]:
    nw, nh = pattern_size
    pts = _corners_xy(corners)
    row_rms, col_rms = [], []
    for r in range(nh):
        seg = pts[r * nw : (r + 1) * nw]
        rm, _ = _line_fit_rms_max(seg)
        row_rms.append(rm)
    for c in range(nw):
        seg = pts[c :: nw]
        rm, _ = _line_fit_rms_max(seg)
        col_rms.append(rm)
    return {
        "row_rms_mean": float(np.mean(row_rms)),
        "col_rms_mean": float(np.mean(col_rms)),
    }


def _warp_magnitude_and_radius(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    wh: Tuple[int, int],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = wh
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_k,
        (w, h),
        cv2.CV_32FC1,
    )
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx = map1.astype(np.float32) - xg
    dy = map2.astype(np.float32) - yg
    mag = np.sqrt(dx * dx + dy * dy)
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    r = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    return mag, r


def plot_warp_vs_radius(mag: np.ndarray, r: np.ndarray, out_path: str, bins: int = 48) -> None:
    """畸变校正引起的采样位移模长，随距主点距离的变化（分箱均值）。"""
    flat_m = mag.ravel()
    flat_r = r.ravel()
    r_max = float(flat_r.max())
    if r_max < 1e-6:
        return
    edges = np.linspace(0, r_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    stds = []
    for i in range(bins):
        m = (flat_r >= edges[i]) & (flat_r < edges[i + 1])
        if i == bins - 1:
            m = flat_r >= edges[i]
        seg = flat_m[m]
        if seg.size == 0:
            means.append(np.nan)
            stds.append(np.nan)
        else:
            means.append(float(np.mean(seg)))
            stds.append(float(np.std(seg)))
    means = np.array(means)
    stds = np.nan_to_num(stds, nan=0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(centers, means, color="#1f77b4", lw=2, label="分箱平均 |位移|")
    ax.fill_between(
        centers,
        np.maximum(0, means - stds),
        means + stds,
        color="#1f77b4",
        alpha=0.2,
        label="±1 标准差",
    )
    ax.set_xlabel("距主点的距离 r（像素）")
    ax.set_ylabel("采样位移模长 |Δ|（像素）")
    ax.set_title("去畸变时像素采样偏移 vs 径向距离（边缘通常更大）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_warp_histogram(mag: np.ndarray, out_path: str, max_samples: int = 200000) -> None:
    m = mag.ravel()
    if m.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(m.size, size=max_samples, replace=False)
        m = m[idx]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.hist(m, bins=80, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("采样位移模长 |Δ|（像素）")
    ax.set_ylabel("像素个数")
    ax.set_title("全图位移模长分布（黑边处常为 0，可忽略峰值）")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_view_rms(npz_path: str, image_dir: str, out_path: str) -> None:
    data = np.load(npz_path)
    if "per_view_errors" not in data.files:
        return
    err = np.asarray(data["per_view_errors"]).ravel()
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    )
    names = [os.path.basename(p) for p in images[: len(err)]]
    if len(names) != len(err):
        names = [f"#{i}" for i in range(len(err))]

    order = np.argsort(err)
    err_s = err[order]
    names_s = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(err))), dpi=120)
    y = np.arange(len(err_s))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(err_s)))
    ax.barh(y, err_s, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=8)
    ax.set_xlabel("每张标定图的 RMS 重投影误差（像素）")
    ax.set_title("标定数据集：逐张误差（越低越好）")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_straightness_bars(before: dict, after: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    keys = ["row_rms_mean", "col_rms_mean"]
    labels = ["行间 RMS", "列间 RMS"]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w / 2, [before[k] for k in keys], w, label="去畸变前", color="#ff7f0e")
    ax.bar(x + w / 2, [after[k] for k in keys], w, label="去畸变后", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("像素")
    ax.set_title("棋盘格线直线度（角点到拟合线 RMS）")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="去畸变 / 标定结果曲线分析图")
    p.add_argument(
        "--npz",
        default="calibration_result_v1.npz",
        help="标定 npz（默认 v1；实验新标定可用 calibration_result_latest.npz）",
    )
    p.add_argument("--image-dir", default="calib_images", help="与标定相同的图片目录（用于柱状图文件名）")
    p.add_argument(
        "--sample",
        "-i",
        default="",
        help="可选：一张测试图，用于棋盘直线度前后对比（需能检测到棋盘）",
    )
    p.add_argument("--pattern", nargs=2, type=int, default=[6, 8], metavar=("W", "H"))
    p.add_argument("--alpha", type=float, default=1.0, help="getOptimalNewCameraMatrix 的 alpha")
    p.add_argument("--out-dir", default=".", help="PNG 输出目录")
    args = p.parse_args()

    data = np.load(args.npz)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    wh = tuple(int(x) for x in np.asarray(data["image_size"]).ravel()[:2])

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    mag, r = _warp_magnitude_and_radius(K, dist, wh, args.alpha)
    plot_warp_vs_radius(mag, r, os.path.join(out_dir, "analysis_warp_vs_radius.png"))
    plot_warp_histogram(mag, os.path.join(out_dir, "analysis_warp_histogram.png"))
    plot_per_view_rms(args.npz, args.image_dir, os.path.join(out_dir, "analysis_per_view_rms.png"))

    print("已生成：")
    print(f"  {os.path.join(out_dir, 'analysis_warp_vs_radius.png')}  — 位移~半径曲线")
    print(f"  {os.path.join(out_dir, 'analysis_warp_histogram.png')}  — 位移分布直方图")
    print(f"  {os.path.join(out_dir, 'analysis_per_view_rms.png')}  — 各张标定图 RMS 柱状图")

    if args.sample:
        img = cv2.imread(args.sample)
        if img is None:
            print("未读取 --sample，跳过棋盘直线度对比图")
            return
        h, w = img.shape[:2]
        if (w, h) != wh:
            print(f"警告：sample 尺寸 {(w,h)} 与 npz 中 image_size {wh} 不一致，仍尝试检测")
        new_k, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_k, (w, h), cv2.CV_16SC2)
        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        pattern_size = (int(args.pattern[0]), int(args.pattern[1]))
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        go, co = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern_size, flags)
        gu, cu = cv2.findChessboardCorners(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), pattern_size, flags)
        if go and gu:
            co = cv2.cornerSubPix(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), co, (11, 11), (-1, -1), crit
            )
            cu = cv2.cornerSubPix(
                cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), cu, (11, 11), (-1, -1), crit
            )
            sb = _chessboard_straightness(co, pattern_size)
            sa = _chessboard_straightness(cu, pattern_size)
            plot_straightness_bars(sb, sa, os.path.join(out_dir, "analysis_straightness_compare.png"))
            print(f"  {os.path.join(out_dir, 'analysis_straightness_compare.png')}  — 棋盘直线度前后对比")
        else:
            print("样本图未检测到棋盘，未生成 analysis_straightness_compare.png")


if __name__ == "__main__":
    main()
