import cv2
import torch
import supervision as sv
import numpy as np

# =========================
# COCO model
# =========================
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

# =========================
# ByteTrack (안정 버전)
# =========================
tracker = sv.ByteTrack()

VIDEO_PATH = r"C:\project\crosswalk-traffic-light-detection-yolov5\videos\yellow_moving_car_3.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

VALID_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # =========================
    # RAW CHECK (YOLO 정상 여부)
    # =========================
    print("RAW DETECTIONS:", len(results.xyxy[0]))

    # =========================
    # 1. YOLO → Detections 변환 (중요)
    # =========================
    detections_list = []

    for *xyxy, conf, cls in results.xyxy[0]:
        cls = int(cls)

        if cls in VALID_CLASSES:
            x1, y1, x2, y2 = map(float, xyxy)

            detections_list.append([
                x1, y1, x2, y2,
                float(conf),
                cls
            ])

    # =========================
    # 2. numpy로 변환 (ByteTrack 안정 입력)
    # =========================
    if len(detections_list) > 0:
        dets = np.array(detections_list)

        detections = sv.Detections(
            xyxy=dets[:, :4],
            confidence=dets[:, 4],
            class_id=dets[:, 5].astype(int)
        )
    else:
        detections = sv.Detections.empty()

    print("AFTER FILTER:", len(detections))

    # =========================
    # 3. TRACKING
    # =========================
    tracked = tracker.update_with_detections(detections)

    print("TRACKED:", len(tracked))

    # =========================
    # 4. DRAW TRACKING RESULT
    # =========================
    for xyxy, cls, tid in zip(
        tracked.xyxy,
        tracked.class_id,
        tracked.tracker_id
    ):
        x1, y1, x2, y2 = map(int, xyxy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{model.names[int(cls)]} ID:{tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # =========================
    # SHOW
    # =========================
    cv2.imshow("COCO + ByteTrack FIXED", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()