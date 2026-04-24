import argparse
from typing import Dict, Tuple

import cv2
import numpy as np


def warp_magnitude_bgr(map1, map2, w: int, h: int) -> np.ndarray:
    """initUndistortRectifyMap 的采样坐标相对理想网格的偏移模长，边缘通常更大。"""
    map1f, map2f = cv2.convertMaps(map1, map2, cv2.CV_32FC1)
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(x, y)
    dx = map1f.astype(np.float32) - xg
    dy = map2f.astype(np.float32) - yg
    mag = np.sqrt(dx * dx + dy * dy)
    mmax = float(mag.max()) if mag.size else 0.0
    if mmax < 1e-6:
        norm = np.zeros_like(mag, dtype=np.uint8)
    else:
        norm = np.clip(mag * (255.0 / mmax), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO), mmax


def draw_displacement_arrows(
    bgr: np.ndarray,
    map1,
    map2,
    step: int = 48,
    scale: float = 1.0,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """在规则网格上绘制采样位移箭头（方向=采样方向，长度=位移大小）。"""
    out = bgr.copy()
    map1f, map2f = cv2.convertMaps(map1, map2, cv2.CV_32FC1)
    h, w = out.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            sx = float(map1f[y, x])
            sy = float(map2f[y, x])
            dx = (sx - x) * scale
            dy = (sy - y) * scale
            p0 = (int(x), int(y))
            p1 = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(out, p0, p1, color, 1, cv2.LINE_AA, tipLength=0.25)
    return out


CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)


def _corners_xy(corners: np.ndarray) -> np.ndarray:
    return corners.reshape(-1, 2).astype(np.float64)


def line_fit_rms_max(pts: np.ndarray) -> Tuple[float, float]:
    """点到同一条拟合直线的距离：RMS 与最大偏差（像素）。"""
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


def chessboard_straightness(corners: np.ndarray, pattern_size: Tuple[int, int]) -> Dict[str, float]:
    """行间/列内角点相对拟合直线的偏差，作为「格线弯曲」的简单数值。"""
    nw, nh = pattern_size
    pts = _corners_xy(corners)
    row_rms, row_max, col_rms, col_max = [], [], [], []
    for r in range(nh):
        seg = pts[r * nw : (r + 1) * nw]
        rm, xm = line_fit_rms_max(seg)
        row_rms.append(rm)
        row_max.append(xm)
    for c in range(nw):
        seg = pts[c :: nw]
        rm, xm = line_fit_rms_max(seg)
        col_rms.append(rm)
        col_max.append(xm)
    return {
        "row_rms_mean": float(np.mean(row_rms)),
        "row_rms_max": float(np.max(row_rms)),
        "row_pt_max": float(np.max(row_max)),
        "col_rms_mean": float(np.mean(col_rms)),
        "col_rms_max": float(np.max(col_rms)),
        "col_pt_max": float(np.max(col_max)),
    }


def grid_directions(pts: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """棋盘两族边在图像平面内的平均方向（单位向量）。"""
    nw, nh = pattern_size
    d_rows = []
    for r in range(nh):
        row = pts[r * nw : (r + 1) * nw]
        for c in range(nw - 1):
            d_rows.append(row[c + 1] - row[c])
    d_cols = []
    for c in range(nw):
        for r in range(nh - 1):
            d_cols.append(pts[(r + 1) * nw + c] - pts[r * nw + c])
    dr = np.mean(d_rows, axis=0) if d_rows else np.array([1.0, 0.0])
    dc = np.mean(d_cols, axis=0) if d_cols else np.array([0.0, 1.0])
    dr = dr / (np.linalg.norm(dr) + 1e-12)
    dc = dc / (np.linalg.norm(dc) + 1e-12)
    return dr.astype(np.float64), dc.astype(np.float64)


def draw_parallel_families(
    bgr: np.ndarray,
    center: np.ndarray,
    d: np.ndarray,
    color: tuple[int, int, int],
    step: float,
    half_lines: int = 4,
) -> None:
    """沿 center 两侧、垂直于 d 的方向平移，画与 d 平行的一组直线。"""
    h, w_img = bgr.shape[:2]
    d = d / (np.linalg.norm(d) + 1e-12)
    perp = np.array([-d[1], d[0]], dtype=np.float64)
    L = float(max(w_img, h) * 2)
    rect = (0, 0, w_img, h)
    for k in list(range(-half_lines, 0)) + list(range(1, half_lines + 1)):
        base = center + k * step * perp
        p1 = (base - L * d).astype(np.float64)
        p2 = (base + L * d).astype(np.float64)
        ok, q1, q2 = cv2.clipLine(
            rect, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))
        )
        if ok:
            cv2.line(bgr, q1, q2, color, 1, cv2.LINE_AA)


def overlay_chessboard_reference_lines(
    bgr: np.ndarray,
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
) -> None:
    """在图上画与棋盘两族边平行的参考线（穿过棋盘区域）。"""
    pts = _corners_xy(corners)
    center = np.mean(pts, axis=0)
    dr, dc = grid_directions(pts, pattern_size)
    nw, nh = pattern_size
    # 相邻格线间距（像素量级），用于平行线疏密
    pr = np.array([-dr[1], dr[0]], dtype=np.float64)
    pc = np.array([-dc[1], dc[0]], dtype=np.float64)
    step_r = 1.0
    if nh > 1:
        mid0 = np.mean(pts[0:nw], axis=0)
        mid1 = np.mean(pts[nw : 2 * nw], axis=0)
        step_r = max(12.0, abs(np.dot(mid1 - mid0, pr)) * 0.45)
    step_c = 1.0
    if nw > 1:
        c0 = np.mean(pts[0::nw], axis=0)
        c1 = np.mean(pts[1::nw], axis=0)
        step_c = max(12.0, abs(np.dot(c1 - c0, pc)) * 0.45)
    # 主方向稍粗，便于辨认
    h, w_img = bgr.shape[:2]
    L = float(max(w_img, h) * 2)
    rect = (0, 0, w_img, h)
    for d, col in ((dr, (0, 255, 255)), (dc, (0, 165, 255))):
        base = center.astype(np.float64)
        p1 = (base - L * d).astype(np.float64)
        p2 = (base + L * d).astype(np.float64)
        ok, q1, q2 = cv2.clipLine(
            rect, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1])))
        )
        if ok:
            cv2.line(bgr, q1, q2, col, 2, cv2.LINE_AA)
    draw_parallel_families(bgr, center, dr, (0, 220, 255), step_r)
    draw_parallel_families(bgr, center, dc, (0, 140, 255), step_c)


def draw_chessboard_lines(
    bgr: np.ndarray,
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """按行列连接棋盘角点，直观看行列线形状。"""
    pts = _corners_xy(corners)
    nw, nh = pattern_size
    for r in range(nh):
        row = pts[r * nw : (r + 1) * nw].astype(np.int32)
        cv2.polylines(bgr, [row.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    for c in range(nw):
        col = pts[c::nw].astype(np.int32)
        cv2.polylines(bgr, [col.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def corner_zoom_strip(
    orig: np.ndarray,
    und: np.ndarray,
    patch: int = 160,
    zoom: int = 3,
) -> np.ndarray:
    """四角 + 中心：每行 [原图放大 | 去畸变放大]，畸变在边角最明显。"""
    h, w = orig.shape[:2]
    patch = min(patch, w // 2, h // 2)
    regions = [
        ("TL", 0, 0),
        ("TR", w - patch, 0),
        ("BL", 0, h - patch),
        ("BR", w - patch, h - patch),
        ("C ", (w - patch) // 2, (h - patch) // 2),
    ]
    rows = []
    for name, x0, y0 in regions:
        o = orig[y0 : y0 + patch, x0 : x0 + patch]
        u = und[y0 : y0 + patch, x0 : x0 + patch]
        o_big = cv2.resize(o, (o.shape[1] * zoom, o.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
        u_big = cv2.resize(u, (u.shape[1] * zoom, u.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([o_big, np.zeros((o_big.shape[0], 6, 3), dtype=np.uint8), u_big])
        cv2.putText(pair, name.strip(), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        rows.append(pair)
    gap = np.zeros((8, rows[0].shape[1], 3), dtype=np.uint8)
    gap[:] = (30, 30, 30)
    out = rows[0]
    for r in rows[1:]:
        out = np.vstack([out, gap, r])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="用标定 npz 对单张图去畸变")
    p.add_argument(
        "--npz",
        default="calibration_result_v1.npz",
        help="标定结果文件（默认稳定版 v1，实验可用 calibration_result_latest.npz）",
    )
    p.add_argument(
        "--input",
        "-i",
        default="calib_images/right10.jpg",
        help="输入图片路径",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="getOptimalNewCameraMatrix 的 alpha：0=裁掉全部无效像素（视野略小），"
        "1=保留全部原图视野（边缘可能有黑边）。常用 0.5~1.0",
    )
    p.add_argument(
        "--interp",
        choices=("linear", "cubic", "lanczos4"),
        default="linear",
        help="remap 插值：linear 快；cubic/lanczos4 边缘略锐、更慢",
    )
    p.add_argument(
        "--no-compare",
        action="store_true",
        help="不写出并排对比图 undistort_compare.jpg",
    )
    p.add_argument(
        "--no-effect-viz",
        action="store_true",
        help="不写出去畸变效果分析图（位移热力图、叠影、边角放大）",
    )
    p.add_argument(
        "--pattern",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=[6, 8],
        help="棋盘内角点列数、行数，须与 camera_calibration.py 中 pattern_size 一致（默认 6 8）",
    )
    p.add_argument(
        "--no-board-overlay",
        action="store_true",
        help="不检测棋盘、不画参考线、不打印直线度数值",
    )
    args = p.parse_args()

    interp_map = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp = interp_map[args.interp]

    data = np.load(args.npz)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    rms_repro = float(data["rms_reprojection"]) if "rms_reprojection" in data.files else None
    mean_err = float(data["mean_error"]) if "mean_error" in data.files else None
    calib_flags = int(data["calib_flags"]) if "calib_flags" in data.files else None

    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"测试图片没读到：{args.input}")

    h, w = img.shape[:2]
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])

    # 焦距比偏离 1 太多时，标定可能不稳；去畸变后直线度仍会受限于标定质量
    f_ratio = fx / fy if fy > 0 else 1.0
    print("===== 标定诊断（只读） =====")
    print(f"  fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    print(f"  fx/fy = {f_ratio:.4f}（接近 1 更典型；偏离大时检查棋盘格参数与多角度覆盖）")
    if calib_flags is not None:
        has_rat = (calib_flags & cv2.CALIB_RATIONAL_MODEL) != 0
        print(f"  标定 flags: {calib_flags}（CALIB_RATIONAL_MODEL={'是' if has_rat else '否'}）")
    if rms_repro is not None:
        print(f"  总体 RMS 重投影 (calibrateCamera): {rms_repro:.3f} px")
    if mean_err is not None:
        print(f"  各视角 RMS 的平均: {mean_err:.3f} px")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), args.alpha, (w, h)
    )
    x, y, rw, rh = roi
    roi_area = rw * rh
    full_area = w * h
    print(f"\n===== 去畸变（alpha={args.alpha}） =====")
    print(f"  有效 ROI: {rw}x{rh} @ ({x},{y})，约占画面 {100.0 * roi_area / full_area:.1f}%")

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (w, h),
        cv2.CV_16SC2,
    )
    undistorted = cv2.remap(img, map1, map2, interpolation=interp, borderMode=cv2.BORDER_CONSTANT)

    undistorted_crop = undistorted[y : y + rh, x : x + rw]

    cv2.imwrite("original_test.jpg", img)
    cv2.imwrite("undistorted.jpg", undistorted)
    cv2.imwrite("undistorted_crop.jpg", undistorted_crop)

    print("\n已保存：")
    print("  original_test.jpg")
    print("  undistorted.jpg")
    print("  undistorted_crop.jpg")

    pattern_size = (int(args.pattern[0]), int(args.pattern[1]))
    if not args.no_board_overlay:
        gray_o = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_u = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        ok_o, co = cv2.findChessboardCorners(gray_o, pattern_size, CHESSBOARD_FLAGS)
        ok_u, cu = cv2.findChessboardCorners(gray_u, pattern_size, CHESSBOARD_FLAGS)
        print("\n===== 棋盘：平行参考线 + 直线度（格线弯曲） =====")
        print("  说明：行间/列间 RMS = 该排角点到一条拟合直线的距离 RMS（像素）；")
        print("        去畸变后通常应略降。黄/橙线 = 与棋盘两族边平行的参考线。")
        if ok_o:
            co = cv2.cornerSubPix(gray_o, co, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            so = chessboard_straightness(co, pattern_size)
            print(
                f"  [原图] 行间 RMS 均值={so['row_rms_mean']:.3f} 最大={so['row_rms_max']:.3f} | "
                f"单点最大={so['row_pt_max']:.3f}"
            )
            print(
                f"         列间 RMS 均值={so['col_rms_mean']:.3f} 最大={so['col_rms_max']:.3f} | "
                f"单点最大={so['col_pt_max']:.3f}"
            )
        else:
            print("  [原图] 未检测到棋盘角点（检查 --pattern 或换图）")
        if ok_u:
            cu = cv2.cornerSubPix(gray_u, cu, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            su = chessboard_straightness(cu, pattern_size)
            print(
                f"  [去畸变] 行间 RMS 均值={su['row_rms_mean']:.3f} 最大={su['row_rms_max']:.3f} | "
                f"单点最大={su['row_pt_max']:.3f}"
            )
            print(
                f"            列间 RMS 均值={su['col_rms_mean']:.3f} 最大={su['col_rms_max']:.3f} | "
                f"单点最大={su['col_pt_max']:.3f}"
            )
        else:
            print("  [去畸变] 未检测到棋盘角点（可略调对比度或 --pattern）")
        print(
            "  注：去畸变后 RMS 不保证总变小（插值平滑、角点检测噪声、透视都会影响）；"
            "请与 undistort_grid_lines.jpg 中参考线目视对照。"
        )
        if ok_o and ok_u:
            line_o = img.copy()
            line_u = undistorted.copy()
            overlay_chessboard_reference_lines(line_o, co, pattern_size)
            overlay_chessboard_reference_lines(line_u, cu, pattern_size)
            cv2.putText(
                line_o,
                "orig (cyan+orange = board dirs)",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                line_u,
                "undistorted",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            gap = np.zeros((h, 8, 3), dtype=np.uint8)
            gap[:] = (40, 40, 40)
            grid_cmp = np.hstack([line_o, gap, line_u])
            cv2.imwrite("undistort_grid_lines.jpg", grid_cmp)
            print("  已保存 undistort_grid_lines.jpg（左原图 / 右去畸变，叠加与棋盘边平行的线族）")

            # 双色叠加：原图角点线(橙) + 去畸变角点线(青)
            board_overlay = cv2.addWeighted(img, 0.65, undistorted, 0.35, 0)
            draw_chessboard_lines(board_overlay, co, pattern_size, (0, 165, 255), 1)
            draw_chessboard_lines(board_overlay, cu, pattern_size, (255, 255, 0), 1)
            cv2.putText(
                board_overlay,
                "orange=orig board lines, cyan=undistorted board lines",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite("undistort_board_lines_overlay.jpg", board_overlay)
            print("  已保存 undistort_board_lines_overlay.jpg（橙=原图角点线，青=去畸变角点线）")

    if not args.no_compare:
        # 缩放到同高并排，便于肉眼看直线/边缘
        target_h = min(h, 720)
        scale = target_h / float(h)
        tw = max(1, int(w * scale))
        a = cv2.resize(img, (tw, target_h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(undistorted, (tw, target_h), interpolation=cv2.INTER_AREA)
        gap = np.zeros((target_h, 8, 3), dtype=np.uint8)
        gap[:] = (40, 40, 40)
        compare = np.hstack([a, gap, b])
        cv2.putText(
            compare,
            "original",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            compare,
            "undistorted",
            (tw + 20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite("undistort_compare.jpg", compare)
        print("  undistort_compare.jpg（左原图 / 右去畸变）")

    if not args.no_effect_viz:
        warp_vis, mmax = warp_magnitude_bgr(map1, map2, w, h)
        cv2.putText(
            warp_vis,
            f"max shift {mmax:.1f} px (src sampling offset)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite("undistort_warp_heatmap.jpg", warp_vis)
        arrows = draw_displacement_arrows(img, map1, map2, step=48, scale=1.0, color=(0, 255, 255))
        cv2.putText(
            arrows,
            "displacement arrows: dir=sampling direction, len=shift",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite("undistort_displacement_arrows.jpg", arrows)
        blend = cv2.addWeighted(img, 0.5, undistorted, 0.5, 0)
        cv2.imwrite("undistort_blend50.jpg", blend)
        zoom_strip = corner_zoom_strip(img, undistorted)
        cv2.imwrite("undistort_corner_zoom.jpg", zoom_strip)
        print("  undistort_warp_heatmap.jpg（畸变校正时像素采样偏移，边角颜色更亮=形变更大）")
        print("  undistort_displacement_arrows.jpg（网格箭头：方向=采样方向，长度=位移像素）")
        print("  undistort_blend50.jpg（原图与去畸变 50% 叠影，重影处=几何不一致）")
        print("  undistort_corner_zoom.jpg（四角+中心放大：直墙/棋盘线更易对比）")
        print(
            "\n提示：手机广角「桶形」主要在画面边缘；看边角放大或热力图比整图并排更敏感。"
            "若叠影仍很淡，说明该镜头畸变弱或标定已把大部分形变消掉。"
        )


if __name__ == "__main__":
    main()
