import argparse
import cv2
import glob
import os

import numpy as np

# ========= 参数 =========
image_dir = "calib_images"
# 内角点数 (每行、每列的内角个数)，须与实物一致；常见错误是把方格数当成内角数
pattern_size = (6, 8)
square_size = 1.0

# 标定优化迭代（与角点 subpix 分开）
calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# 标定标志：理性模型 (k1,k2,p1,p2,k3,k4,k5,k6)，与 initUndistortRectifyMap / undistort 兼容
CALIB_FLAGS = int(cv2.CALIB_RATIONAL_MODEL)

# 棋盘角点检测：自适应阈值 + 归一化，减轻反光、光照不均的影响
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# 亚像素优化停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def main(output_npz: str) -> None:
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    image_names = []

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))
    )

    if len(images) == 0:
        raise ValueError("calib_images 文件夹里没有图片")

    img_size = None
    success_count = 0

    corners_dir = os.path.join(image_dir, "corners_result")
    os.makedirs(corners_dir, exist_ok=True)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, chessboard_flags)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            image_names.append(os.path.basename(fname))
            success_count += 1

            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners2, ret)

            save_path = os.path.join(corners_dir, "corners_" + os.path.basename(fname))
            cv2.imwrite(save_path, vis)

            print(f"[成功] {os.path.basename(fname)}")
        else:
            print(f"[失败] {os.path.basename(fname)}")

    print(f"\n成功检测角点: {success_count}/{len(images)}")

    if success_count < 5:
        raise ValueError("成功图片太少，先检查 pattern_size 是否写对")

    (
        rms_reprojection,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        std_dev_intrinsics,
        std_dev_extrinsics,
        per_view_errors,
    ) = cv2.calibrateCameraExtended(
        objpoints,
        imgpoints,
        img_size,
        None,
        None,
        flags=CALIB_FLAGS,
        criteria=calib_criteria,
    )

    per_view_errors = np.asarray(per_view_errors).reshape(-1)
    mean_per_view_rms = float(np.mean(per_view_errors))
    std_per_view = float(np.std(per_view_errors))
    min_idx = int(np.argmin(per_view_errors))
    max_idx = int(np.argmax(per_view_errors))

    print("\n===== 标定结果 =====")
    print(f"CALIB_FLAGS = {CALIB_FLAGS} (含 CALIB_RATIONAL_MODEL={cv2.CALIB_RATIONAL_MODEL})")
    print(f"图像尺寸 (w, h): {img_size}")
    print(f"总体 RMS 重投影误差 (calibrateCamera 返回值): {rms_reprojection:.6f} px")
    print(f"各视角 RMS: 均值={mean_per_view_rms:.6f} px, 标准差={std_per_view:.6f} px")
    print(f"  最小: {per_view_errors[min_idx]:.6f} px  [{image_names[min_idx]}]")
    print(f"  最大: {per_view_errors[max_idx]:.6f} px  [{image_names[max_idx]}]")
    print("\n逐张 RMS (像素):")
    for name, err in zip(image_names, per_view_errors):
        print(f"  {err:8.4f}  {name}")

    print("\nCamera matrix:\n", camera_matrix)
    print("Distortion coefficients (长度随模型而定，与 undistort 一致即可):\n", dist_coeffs)

    if std_dev_intrinsics is not None and std_dev_intrinsics.size:
        sdi = np.asarray(std_dev_intrinsics).reshape(-1)
        print(
            "\n内参标准差 (前 4 项通常对应 fx, fy, cx, cy 的不确定度，具体顺序见 OpenCV 文档):"
        )
        print(" ", sdi[: min(6, len(sdi))])

    mean_error = mean_per_view_rms

    np.savez(
        output_npz,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        mean_error=mean_error,
        rms_reprojection=rms_reprojection,
        per_view_errors=per_view_errors,
        calib_flags=np.int32(CALIB_FLAGS),
        image_size=np.array([img_size[0], img_size[1]], dtype=np.int32),
    )

    print(f"\n参数已保存到 {output_npz}")
    print("  字段: camera_matrix, dist_coeffs, mean_error, rms_reprojection, per_view_errors, calib_flags, image_size")
    if output_npz != "calibration_result_v1.npz":
        print("  提示: 稳定存档请使用 calibration_result_v1.npz；满意后可手动复制覆盖该文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="棋盘格相机标定")
    parser.add_argument(
        "-o",
        "--output",
        default="calibration_result_latest.npz",
        help="输出 npz（默认 calibration_result_latest.npz，不覆盖 v1）",
    )
    args = parser.parse_args()
    if args.output == "calibration_result_v1.npz":
        print("警告: 将写入 calibration_result_v1.npz；请确认有意更新稳定存档。\n")
    main(args.output)
