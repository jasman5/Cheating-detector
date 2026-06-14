import cv2
import mediapipe as mp
import math
import time
import os

# ---------------- INITIALIZE ---------------- #

mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Set bigger camera resolution
cap.set(3, 1280)
cap.set(4, 720)

last_position = None
movement_threshold = 40

warning_count = 0
capture_id = 0

# Create folder for evidence
if not os.path.exists("captures"):
    os.makedirs("captures")

prev_time = 0

# ---------------- LOOP ---------------- #

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape

    # ---------------- DETECTIONS ---------------- #

    face_results = face_detection.process(rgb)
    pose_results = pose.process(rgb)
    mesh_results = face_mesh.process(rgb)

    face_count = 0
    cheating_detected = False

    # ---------------- FACE DETECTION ---------------- #

    if face_results.detections:

        face_count = len(face_results.detections)

        for detection in face_results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            cv2.rectangle(frame, (x, y),
                          (x + width, y + height),
                          (0, 255, 0), 2)

    # ---------------- ALERTS ---------------- #

    if face_count == 0:

        cheating_detected = True

        cv2.putText(frame,
                    "ALERT: No Person Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    elif face_count > 1:

        cheating_detected = True

        cv2.putText(frame,
                    "ALERT: Multiple Persons Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3)

    # ---------------- HEAD MOVEMENT ---------------- #

    if pose_results.pose_landmarks:

        nose = pose_results.pose_landmarks.landmark[0]

        cx = int(nose.x * w)
        cy = int(nose.y * h)

        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        if last_position is not None:

            dx = abs(cx - last_position[0])
            dy = abs(cy - last_position[1])

            if dx > movement_threshold or dy > movement_threshold:

                cheating_detected = True

                cv2.putText(frame,
                            "Suspicious Head Movement",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            3)

        last_position = (cx, cy)

    # ---------------- LOOKING AWAY DETECTION ---------------- #

    if mesh_results.multi_face_landmarks:

        for face_landmarks in mesh_results.multi_face_landmarks:

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]

            left_x = int(left_eye.x * w)
            right_x = int(right_eye.x * w)
            nose_x = int(nose_tip.x * w)

            eye_center = (left_x + right_x) // 2

            difference = abs(nose_x - eye_center)

            # Looking sideways
            if difference > 25:

                cheating_detected = True

                cv2.putText(frame,
                            "Looking Away Detected",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255),
                            3)

    # ---------------- SAVE SCREENSHOT ---------------- #

    if cheating_detected:

        warning_count += 1

        filename = f"captures/cheating_{capture_id}.jpg"

        cv2.imwrite(filename, frame)

        capture_id += 1

    # ---------------- WARNING COUNT ---------------- #

    cv2.putText(frame,
                f"Warnings: {warning_count}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3)

    # ---------------- FPS ---------------- #

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3)

    # ---------------- SHOW WINDOW ---------------- #

    cv2.namedWindow("Cheating Detector", cv2.WINDOW_NORMAL)

    cv2.imshow("Cheating Detector", frame)

    # Press Q to quit
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ---------------- RELEASE ---------------- #

cap.release()
cv2.destroyAllWindows()