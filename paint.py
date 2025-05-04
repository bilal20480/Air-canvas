import cv2
import numpy as np
import random
import string
import mediapipe as mp
import time

class PaintApp:
    def __init__(self):
        self.drawing = False  # True if drawing gesture is detected
        self.brush_color = (255, 0, 0)  # Default color: blue
        self.brush_size = 10
        self.last_x, self.last_y = -1, -1
        self.colors = [(255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
        self.selected_color_index = 0
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.canvas = None
        self.target = self.generate_target()
        self.feedback = ""
        self.start_time = time.time()

    def generate_target(self):
        target_type = random.choice(['letter', 'number', 'shape'])
        if target_type == 'letter':
            return random.choice(string.ascii_uppercase)
        elif target_type == 'number':
            return str(random.randint(0, 9))
        else:
            return random.choice(['circle', 'triangle', 'square'])

    def draw_target(self, frame):
        target = self.target
        font = cv2.FONT_HERSHEY_SIMPLEX
        target_x = frame.shape[1] - 200
        target_y = 50
        if target.isdigit() or target.isalpha():
            cv2.putText(frame, target, (target_x, target_y), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            if target == 'circle':
                cv2.circle(frame, (frame.shape[1] - 150, 100), 50, (0, 255, 0), 2)
            elif target == 'triangle':
                points = np.array([[frame.shape[1] - 150, 100], [frame.shape[1] - 200, 200], [frame.shape[1] - 100, 200]], np.int32)
                cv2.polylines(frame, [points.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            elif target == 'square':
                cv2.rectangle(frame, (frame.shape[1] - 200, 50), (frame.shape[1] - 100, 150), (0, 255, 0), 2)

    def draw_color_palette(self, frame):
        for i, color in enumerate(self.colors):
            x1, y1 = 50 + i * 100, 10
            x2, y2 = x1 + 80, 90
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            if i == self.selected_color_index:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

    def process_hand_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                # Check if both index and middle fingers are extended
                index_extended = index_tip.y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_extended = middle_tip.y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                self.drawing = index_extended and middle_extended

                # Change color if finger is on the color palette
                for i in range(len(self.colors)):
                    x1, y1 = 50 + i * 100, 10
                    x2, y2 = x1 + 80, 90
                    if x1 < x < x2 and y1 < y < y2:
                        self.selected_color_index = i
                        self.brush_color = self.colors[i]

                # Draw on the canvas if both fingers are extended
                if self.drawing:
                    if self.last_x != -1 and self.last_y != -1:
                        cv2.line(self.canvas, (self.last_x, self.last_y), (x, y), self.brush_color, self.brush_size)
                    self.last_x, self.last_y = x, y
                else:
                    self.last_x, self.last_y = -1, -1
        else:
            self.drawing = False
            self.last_x, self.last_y = -1, -1

    def check_drawing(self):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(max_contour)

            # Ignore small contours (noise)
            if contour_area < 500:
                self.feedback = "No shape detected!"
                return

            # Create an ideal shape for comparison
            ideal_shape = None
            if self.target == 'circle':
                ideal_shape = self.create_ideal_circle()
            elif self.target == 'square':
                ideal_shape = self.create_ideal_square()
            elif self.target == 'triangle':
                ideal_shape = self.create_ideal_triangle()

            if ideal_shape is not None:
                # Compare the drawn shape with the ideal shape
                similarity = self.compare_shapes(max_contour, ideal_shape)
                if similarity > 0.7:  # Adjust this threshold as needed
                    self.feedback = "Correct!"
                else:
                    self.feedback = "Incorrect, try again!"
            else:
                self.feedback = "Incorrect, try again!"
        else:
            self.feedback = "No shape detected!"

    def create_ideal_circle(self):
        # Create an ideal circle
        ideal_circle = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(ideal_circle, (100, 100), 90, 255, -1)
        contours, _ = cv2.findContours(ideal_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]

    def create_ideal_square(self):
        # Create an ideal square
        ideal_square = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(ideal_square, (20, 20), (180, 180), 255, -1)
        contours, _ = cv2.findContours(ideal_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]

    def create_ideal_triangle(self):
        # Create an ideal triangle
        ideal_triangle = np.zeros((200, 200), dtype=np.uint8)
        points = np.array([[100, 20], [20, 180], [180, 180]], np.int32)
        cv2.fillPoly(ideal_triangle, [points], 255)
        contours, _ = cv2.findContours(ideal_triangle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]

    def compare_shapes(self, contour1, contour2):
        # Compare two contours using Hu Moments
        hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
        hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()
        return cv2.matchShapes(hu1, hu2, cv2.CONTOURS_MATCH_I1, 0)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera not accessible")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            self.draw_color_palette(frame)
            self.draw_target(frame)
            self.process_hand_gesture(frame)
            elapsed_time = int(time.time() - self.start_time)
            if elapsed_time >= 20:
                self.check_drawing()
                self.start_time = time.time()
                self.target = self.generate_target()
                self.canvas = np.zeros_like(frame)  # Clear the canvas for the next target
            cv2.putText(frame, self.feedback, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Time: {20 - elapsed_time}s", (frame.shape[1] - 200, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
            cv2.imshow("PaintApp", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PaintApp()
    app.run()