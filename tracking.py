import cv2
import torch
import supervision as sv

# COCO 모델
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

# ByteTrack
tracker = sv.ByteTrack()

cap = cv2.VideoCapture( 
    r"C:\project\crosswalk-traffic-light-detection-yolov5\videos\yellow_moving_car_3.mp4"
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    print("검출 개수:", len(results.xyxy[0]))

    for *xyxy, conf, cls in results.xyxy[0]:

        x1, y1, x2, y2 = map(int, xyxy)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{int(cls)} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    detections = sv.Detections.from_ultralytics(results)

    # 클래스: 사람, 자동차, 오토바이, 버스, 트럭
    mask = []

    for class_id in detections.class_id:

        if class_id in [0, 2, 3, 5, 7]:
            mask.append(True)
        else:
            mask.append(False)

    detections = detections[mask]

    tracked_objects = tracker.update_with_detections(
        detections
    )

    labels = []

    for tracker_id, class_id in zip(
        tracked_objects.tracker_id,
        tracked_objects.class_id
    ):

        labels.append(
            f"ID:{tracker_id} CLS:{class_id}"
        )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame = box_annotator.annotate(
        scene=frame,
        detections=tracked_objects
    )

    frame = label_annotator.annotate(
        scene=frame,
        detections=tracked_objects,
        labels=labels
    )

    cv2.imshow("ByteTrack", frame)

    if cv2.waitKey(1) & 0xFF == 27: ## ESC를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()