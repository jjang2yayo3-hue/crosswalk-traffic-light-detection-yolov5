import cv2
import torch
import numpy as np
import supervision as sv

# =========================
# Models
# =========================
coco_model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

exp4_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='runs/train/exp4/weights/best.pt'
)

exp124_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='runs/train/exp124/weights/best.pt'
)

# =========================
# Tracker
# =========================
tracker = sv.ByteTrack()

VALID_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

VIDEO_PATH = r"C:\project\crosswalk-traffic-light-detection-yolov5\videos\yellow_moving_car_3.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    # =========================
    # 1. COCO + TRACKING
    # =========================
    results = coco_model(frame)

    detections_list = []

    for *xyxy, conf, cls in results.xyxy[0]:
        cls = int(cls)

        if cls in VALID_CLASSES:
            detections_list.append([
                *xyxy,
                float(conf),
                cls
            ])

    if len(detections_list) > 0:
        dets = np.array(detections_list)

        detections = sv.Detections(
            xyxy=dets[:, :4],
            confidence=dets[:, 4],
            class_id=dets[:, 5].astype(int)
        )
    else:
        detections = sv.Detections.empty()

    tracked = tracker.update_with_detections(detections)

    # =========================
    # DRAW TRACKING
    # =========================
    for xyxy, cls, tid in zip(
        tracked.xyxy,
        tracked.class_id,
        tracked.tracker_id
    ):
        x1, y1, x2, y2 = map(int, xyxy)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{coco_model.names[int(cls)]} ID:{tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # =========================
    # 2. exp4 (crosswalk + signal)
    # =========================
    exp4_results = exp4_model(frame)

    for *xyxy, conf, cls in exp4_results.xyxy[0]:
        cls = int(cls)
        x1, y1, x2, y2 = map(int, xyxy)

        if cls == 0:
            label = "Crosswalk"
            color = (0, 255, 0)
        elif cls == 1:
            label = "Red Light"
            color = (0, 0, 255)
        elif cls == 2:
            label = "Green Light"
            color = (0, 255, 0)
        else:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # =========================
    # 3. exp124 (yellow crosswalk)
    # =========================
    exp124_results = exp124_model(frame)

    for *xyxy, conf, cls in exp124_results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            "Yellow Crosswalk",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # =========================
    # SHOW
    # =========================
    cv2.imshow("Integrated Pipeline", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()