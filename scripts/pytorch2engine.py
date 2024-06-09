from ultralytics import YOLO

model = YOLO("../weights/yolov8/sack/sack-seg-e225.pt")
model.export(format="engine", half=True, device=0, workspace=12)
