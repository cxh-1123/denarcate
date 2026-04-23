# 相机标定与去畸变 — 实验记录（稳定版 v1）

## 稳定参数文件

- **存档（勿被标定脚本覆盖）**：`calibration_result_v1.npz`
- **日常重新标定默认输出**：`calibration_result_latest.npz`（由 `camera_calibration.py` 生成）
- 去畸变 / 分析默认读取 **`calibration_result_v1.npz`**；实验新标定可用 `--npz calibration_result_latest.npz`

---

## v1 标定概况

| 项目 | 内容 |
|------|------|
| 标定图张数 | **18**（`calib_images`，命名 left01–left09、right01–right09） |
| `pattern_size`（内角点列×行） | **(6, 8)**，与 `camera_calibration.py` 中一致 |
| 标定分辨率 `image_size`（宽×高） | **1280 × 1707** |
| 总体 RMS 重投影 | **≈ 1.96 px** |
| 各视角 RMS 均值 | **≈ 1.79 px** |
| 单张误差最低 | **right05.jpg**（≈ **0.81 px**） |
| 单张误差最高（前三） | **right04.jpg**（≈ **3.45 px**）、**left01.jpg**（≈ **3.08 px**）、**left09.jpg**（≈ **2.94 px**） |

---

## 现象与结论（用于项目总结）

1. **去畸变前后「行列直线度」RMS**（棋盘角点相对拟合直线）**没有稳定、明显改善**，部分图去畸变后该指标略升，与插值、角点重检测噪声有关，不宜单独作为「矫正好坏」判据。
2. **主观判断**：当前场景下 **透视（棋盘倾斜、远近）带来的「不共线」往往比强径向畸变更显眼**；去畸变主要校正镜头径向畸变，**不会消除透视**。
3. **fx/fy** 在 v1 中接近 **1**，内参比例合理；后续若换分辨率或换机位，须整组重拍并重标定。

---

## 命令备忘

```text
python camera_calibration.py              # 写入 calibration_result_latest.npz
python undistort_demo.py -i calib_images/right05.jpg   # 默认读 v1
python analyze_undistort.py               # 默认读 v1
python read_npz.py                        # 默认读 v1
```
