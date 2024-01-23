import cv2
import mediapipe as mp
import pyautogui
import math

# Define a class for controlling the computer using vision-based gestures


class VisionOSControl:
    def __init__(self):
        # Initialize the webcam
        self.cam = cv2.VideoCapture(0)

        # Initialize the face mesh and hand tracking models from Mediapipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Get the screen size
        self.screen_width, self.screen_height = pyautogui.size()

    # Process each frame from the webcam
    def process_frame(self, frame):
        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        hand_results = self.hands.process(rgb_frame)
        frame_height, frame_width, _ = frame.shape

        # If hand landmarks are detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Calculate the distance between the thumb tip and index finger tip
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                distance = math.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                print(distance)

                # If the distance is below a threshold, perform a pinch gesture
                if distance < 0.03:
                    print("Pinch gesture detected")
                    pyautogui.click()

        # Process the frame to detect face landmarks
        output = self.face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        # If face landmarks are detected
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                # Get the x and y coordinates of the landmark
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Draw a circle around the landmark
                cv2.circle(frame, (x, y), 3, (0, 255, 0))

                # If it is the second landmark, move the mouse pointer to the corresponding screen position
                if id == 1:
                    screen_x = self.screen_width / frame_width * x
                    screen_y = self.screen_height / frame_height * y
                    pyautogui.moveTo(screen_x, screen_y)

        # Display the frame with annotations
        cv2.imshow("Vision Pointer", frame)

    # Run the vision control loop
    def run(self):
        while True:
            # Read a frame from the webcam
            _, frame = self.cam.read()

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Process the frame
            self.process_frame(frame)

            # If 'q' is pressed, exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all windows
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create an instance of the VisionOSControl class and run the control loop
    vision_control = VisionOSControl()
    vision_control.run()
