from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(description='Process pt file.')
parser.add_argument('--pt_path', help='path to pt file', required=True)
args = parser.parse_args()

model = YOLO(args.pt_path)
model.fuse()
model.info(verbose=False)
model.export(format="onnx", simplify=True)
