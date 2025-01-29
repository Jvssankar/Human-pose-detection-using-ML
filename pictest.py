import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

img_path = "pic.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = pose.process(img_rgb)

annotated_img = img.copy()

if res.pose_landmarks:
    print("Pose landmarks detected!")
    
    for idx, landmark in enumerate(res.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
    
    for landmark in res.pose_landmarks.landmark:
        a, b, c = img.shape
        cx, cy = int(landmark.x * b), int(landmark.y * a)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
    
    mp_drawing.draw_landmarks(annotated_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
else:
    print("No pose landmarks detected!")

cv2.imwrite("Pose_Landmarks.jpg", img)
cv2.imwrite("Pose_Drawing.jpg", annotated_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
pose.close()
