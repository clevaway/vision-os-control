import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize the face mesh model
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize the hand tracking model
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get the screen size
screen_width, screen_height = pyautogui.size()

while True:
    # Read a frame from the webcam
    _, frame = cam.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    hand_results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Get the landmarks for the thumb tip (id 4) and index finger tip (id 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Calculate the distance between the thumb tip and index finger tip
            distance = math.sqrt((thumb_tip.x - index_tip.x)
                                 ** 2 + (thumb_tip.y - index_tip.y)**2)

            print(distance)

            # If the distance is less than a certain threshold, print "Pinch gesture detected"
            if distance < 0.03:
                print("Pinch gesture detected")
                # Perform a click action
                pyautogui.click()

    # Process the frame for face landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_height, frame_width, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:
                # Map the landmark coordinates to the screen coordinates
                screen_x = screen_width / frame_width * x
                screen_y = screen_height / frame_height * y
                pyautogui.moveTo(screen_x, screen_y)

    # Display the frame with annotations
    cv2.imshow("Vision Pointer", frame)

    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
