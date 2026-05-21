import cv2
import mediapipe as mp
import time

# Initialize MediaPipe
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

last_position = None
movement_threshold = 40

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection
    face_results = face_detection.process(rgb)

    # Pose Detection
    pose_results = pose.process(rgb)

    h, w, _ = frame.shape

    # Count faces
    face_count = 0

    if face_results.detections:
        face_count = len(face_results.detections)

        for detection in face_results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Alerts
    if face_count == 0:
        cv2.putText(frame, "ALERT: No Person Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    elif face_count > 1:
        cv2.putText(frame, "ALERT: Multiple Persons Detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Movement Detection using nose landmark
    if pose_results.pose_landmarks:
        nose = pose_results.pose_landmarks.landmark[0]

        cx = int(nose.x * w)
        cy = int(nose.y * h)

        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

        if last_position is not None:
            dx = abs(cx - last_position[0])
            dy = abs(cy - last_position[1])

            if dx > movement_threshold or dy > movement_threshold:
                cv2.putText(frame, "Suspicious Head Movement", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        last_position = (cx, cy)

    cv2.imshow("Cheating Detector", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()