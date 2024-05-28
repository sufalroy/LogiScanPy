from ultralytics import YOLO

model = YOLO("../weights/sack-seg-v2.pt")
model.export(format="engine", half=True, device=0, workspace=12)
