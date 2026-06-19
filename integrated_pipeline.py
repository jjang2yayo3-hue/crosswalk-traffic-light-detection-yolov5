import cv2
import torch
import numpy as np
import supervision as sv
from collections import deque

# Load detection models
coco_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

# Tracker setting
tracker = sv.ByteTrack()

PERSON_CLASS = 0
VEHICLE_CLASSES = [2, 3, 5, 7]
VALID_CLASSES = [PERSON_CLASS] + VEHICLE_CLASSES

VIDEO_PATH = r"C:\project\crosswalk-traffic-light-detection-yolov5\videos\3.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

# Inference speed control
FRAME_SKIP_CUSTOM = 3
frame_count = 0
last_exp4_results = []
last_exp124_results = []

# Recent frame history
HISTORY_LEN = 10
CROSSWALK_MIN_FRAMES = 2
SIGNAL_MIN_FRAMES = 2
CAR_MIN_FRAMES = 1
PERSON_MIN_FRAMES = 1

crosswalk_history = deque(maxlen=HISTORY_LEN)
red_history = deque(maxlen=HISTORY_LEN)
green_history = deque(maxlen=HISTORY_LEN)
moving_car_history = deque(maxlen=HISTORY_LEN)
stopped_car_history = deque(maxlen=HISTORY_LEN)
person_history = deque(maxlen=HISTORY_LEN)

# Vehicle motion history
vehicle_track_history = {}
VEHICLE_HISTORY_LEN = 5
MOVING_SPEED_THRESHOLD = 5.0  # pixel/frame

current_status = "WAITING"
current_risk_score = 0
current_causes = []
last_crosswalk_box = None


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def select_largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=box_area)


def center_of_box(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box_area(box1)
    area2 = box_area(box2)
    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    return inter / union


def make_person_roi(frame_shape):
    h, w = frame_shape[:2]
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.40)
    y2 = int(h * 0.95)
    return (x1, y1, x2, y2)


def is_stable_bool(history, min_frames):
    return sum(history) >= min_frames


def is_crosswalk_sufficient(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box

    box_h = y2 - y1
    enough_height = box_h > h * 0.15
    enough_position = y2 > h * 0.45

    return enough_height or enough_position


def update_vehicle_speed(track_id, point):
    if track_id not in vehicle_track_history:
        vehicle_track_history[track_id] = deque(maxlen=VEHICLE_HISTORY_LEN)

    vehicle_track_history[track_id].append(point)

    points = vehicle_track_history[track_id]

    if len(points) < 2:
        return 0.0

    x_old, y_old = points[0]
    x_new, y_new = points[-1]

    distance = ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5
    speed = distance / (len(points) - 1)

    return speed


def calculate_risk(red_stable, green_stable,
                   moving_car_on_crosswalk,
                   stopped_car_on_crosswalk,
                   approaching_person):
    risk = 0
    causes = []

    signal_unknown = (not red_stable) and (not green_stable)

    if signal_unknown:
        risk += 25
        causes.append("Signal(Not Detected)")

    if red_stable:
        risk += 70
        causes.append("Signal(Red Light)")

    if moving_car_on_crosswalk:
        risk += 70
        causes.append("Moving Car")

    elif stopped_car_on_crosswalk:
        risk += 25
        causes.append("Stopped Car")

    if approaching_person:
        risk += 30
        causes.append("Person")

    if risk >= 70:
        status = "DANGER"
    elif risk >= 25:
        status = "CAUTION"
    else:
        status = "SAFE"

    return status, risk, causes


def draw_status_panel(frame, status, risk_score, causes):
    h, w = frame.shape[:2]

    if status == "SAFE":
        color = (0, 200, 0)
    elif status == "CAUTION":
        color = (0, 200, 255)
    elif status == "DANGER":
        color = (0, 0, 255)
    else:
        color = (180, 180, 180)

    cause_text = ", ".join(causes) if causes else "None"

    panel_w = 330
    panel_h = 85
    panel_x = int((w - panel_w) / 2)
    panel_y = int(h * 0.65)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)

    cv2.putText(frame, status, (panel_x + 15, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.putText(frame, f"Risk: {risk_score}", (panel_x + 15, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"Cause: {cause_text}", (panel_x + 15, panel_y + 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


print("video open =", cap.isOpened())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    run_custom_models = (frame_count == 1) or (frame_count % FRAME_SKIP_CUSTOM == 0)

    annotated = frame.copy()

    tracked_vehicle_infos = []
    tracked_person_boxes = []

    exp4_crosswalk_boxes = []
    yellow_crosswalk_boxes = []

    red_detected = False
    green_detected = False

    # Detect vehicles and pedestrians
    with torch.no_grad():
        results = coco_model(frame)

    detections_list = []

    for *xyxy, conf, cls in results.xyxy[0]:
        cls = int(cls)

        if cls in VALID_CLASSES:
            x1, y1, x2, y2 = map(float, xyxy)
            detections_list.append([x1, y1, x2, y2, float(conf), cls])

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

    active_vehicle_ids = set()

    for xyxy, cls, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        cls = int(cls)
        tid = int(tid)

        if cls == PERSON_CLASS:
            tracked_person_boxes.append((x1, y1, x2, y2))
            color = (255, 0, 255)

        elif cls in VEHICLE_CLASSES:
            vehicle_box = (x1, y1, x2, y2)
            vehicle_bottom_center = ((x1 + x2) / 2, y2)
            vehicle_speed = update_vehicle_speed(tid, vehicle_bottom_center)

            tracked_vehicle_infos.append({
                "box": vehicle_box,
                "track_id": tid,
                "bottom_center": vehicle_bottom_center,
                "speed": vehicle_speed
            })

            active_vehicle_ids.add(tid)
            color = (0, 255, 0)

            cv2.putText(annotated, f"speed:{vehicle_speed:.1f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        else:
            color = (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{coco_model.names[cls]} ID:{tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Clear inactive vehicle IDs
    for tid in list(vehicle_track_history.keys()):
        if tid not in active_vehicle_ids:
            vehicle_track_history.pop(tid, None)

    # Detect crosswalk and signal
    if run_custom_models:
        with torch.no_grad():
            exp4_results = exp4_model(frame)
        last_exp4_results = exp4_results.xyxy[0]

    for *xyxy, conf, cls in last_exp4_results:
        cls = int(cls)
        x1, y1, x2, y2 = map(int, xyxy)

        if cls == 0:
            label = "Crosswalk"
            color = (0, 255, 0)
            exp4_crosswalk_boxes.append((x1, y1, x2, y2))

        elif cls == 1:
            label = "Red Light"
            color = (0, 0, 255)
            red_detected = True

        elif cls == 2:
            label = "Green Light"
            color = (0, 255, 0)
            green_detected = True

        else:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {float(conf):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Detect yellow crosswalk
    if run_custom_models:
        with torch.no_grad():
            exp124_results = exp124_model(frame)
        last_exp124_results = exp124_results.xyxy[0]

    for *xyxy, conf, cls in last_exp124_results:
        x1, y1, x2, y2 = map(int, xyxy)
        yellow_crosswalk_boxes.append((x1, y1, x2, y2))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, f"Yellow Crosswalk {float(conf):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Crosswalk state
    current_crosswalk_boxes = exp4_crosswalk_boxes + yellow_crosswalk_boxes
    current_crosswalk_box = select_largest_box(current_crosswalk_boxes)

    crosswalk_detected = current_crosswalk_box is not None
    crosswalk_history.append(crosswalk_detected)

    if crosswalk_detected:
        last_crosswalk_box = current_crosswalk_box

    crosswalk_stable = is_stable_bool(crosswalk_history, CROSSWALK_MIN_FRAMES)

    crosswalk_visible_enough = (
        last_crosswalk_box is not None and
        is_crosswalk_sufficient(last_crosswalk_box, frame.shape)
    )

    # Signal state
    red_history.append(red_detected)
    green_history.append(green_detected)

    red_stable = is_stable_bool(red_history, SIGNAL_MIN_FRAMES)
    green_stable = is_stable_bool(green_history, SIGNAL_MIN_FRAMES)

    if red_detected and not green_detected:
        red_stable = True
        green_stable = False
    elif green_detected and not red_detected:
        red_stable = False
        green_stable = True

    # Vehicle risk
    moving_car_on_crosswalk_now = False
    stopped_car_on_crosswalk_now = False

    if last_crosswalk_box is not None:
        for vehicle in tracked_vehicle_infos:
            vehicle_bottom_center = vehicle["bottom_center"]
            vehicle_speed = vehicle["speed"]

            if point_in_box(vehicle_bottom_center, last_crosswalk_box):
                if vehicle_speed > MOVING_SPEED_THRESHOLD:
                    moving_car_on_crosswalk_now = True
                else:
                    stopped_car_on_crosswalk_now = True

    moving_car_history.append(moving_car_on_crosswalk_now)
    stopped_car_history.append(stopped_car_on_crosswalk_now)

    moving_car_on_crosswalk_stable = is_stable_bool(moving_car_history, CAR_MIN_FRAMES)
    stopped_car_on_crosswalk_stable = is_stable_bool(stopped_car_history, CAR_MIN_FRAMES)

    # Moving vehicle is more dangerous
    if moving_car_on_crosswalk_stable:
        stopped_car_on_crosswalk_stable = False

    # Person risk
    roi_box = make_person_roi(frame.shape)
    rx1, ry1, rx2, ry2 = roi_box

    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
    cv2.putText(annotated, "Person Risk ROI", (rx1, ry1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    approaching_person_now = False

    for person_box in tracked_person_boxes:
        person_center = center_of_box(person_box)
        center_in_roi = point_in_box(person_center, roi_box)
        person_roi_iou = bbox_iou(person_box, roi_box)

        if center_in_roi or person_roi_iou > 0.05:
            approaching_person_now = True
            break

    person_history.append(approaching_person_now)
    approaching_person_stable = is_stable_bool(person_history, PERSON_MIN_FRAMES)

    # Final risk update
    if crosswalk_stable and crosswalk_visible_enough:
        new_status, new_risk_score, new_causes = calculate_risk(
            red_stable=red_stable,
            green_stable=green_stable,
            moving_car_on_crosswalk=moving_car_on_crosswalk_stable,
            stopped_car_on_crosswalk=stopped_car_on_crosswalk_stable,
            approaching_person=approaching_person_stable
        )

        current_status = new_status
        current_risk_score = new_risk_score
        current_causes = new_causes
    else:
        current_status = "WAITING"
        current_risk_score = 0
        current_causes = []

    draw_status_panel(annotated, current_status, current_risk_score, current_causes)

    cv2.imshow("Integrated Pipeline + Risk Engine", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
