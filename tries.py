import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_width, screen_height = pyautogui.size()

while True:
    _, frame = cam.read()
    # flip the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    # print(landmark_points)
    frame_height, frame_width, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            print(x, y)
            if id == 1:
                screen_x = screen_width / frame_width * x
                screen_y = screen_height / frame_height * y
                pyautogui.moveTo(screen_x, screen_y)

    cv2.imshow("Vison Pointer", frame)
    # Check if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
