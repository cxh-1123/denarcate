import argparse
import numpy as np

parser = argparse.ArgumentParser(description="查看标定 npz 内容")
parser.add_argument(
    "npz",
    nargs="?",
    default="calibration_result_v1.npz",
    help="标定文件路径（默认 calibration_result_v1.npz）",
)
args = parser.parse_args()

data = np.load(args.npz)

print("文件:", args.npz)
print("文件里包含的键：", data.files)

for key in data.files:
    print(f"\n--- {key} ---")
    print(data[key])
