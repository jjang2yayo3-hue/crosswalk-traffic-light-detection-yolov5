import torch

model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='runs/train/exp4/weights/best.pt'
)

print(model.names)