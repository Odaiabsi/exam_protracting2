import cv2
import numpy as np
from face_detector import get_face_detector, find_faces, draw_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

# Load the Haar cascade for sunglasses detection
sunglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')


def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]


def find_eyeball_position(end_points, cx, cy, img_width):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2])
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
    x_distance = abs(cx - img_width / 2)
    y_distance = abs(cy - (end_points[1] + end_points[3]) / 2)
    x_prob = x_distance / (img_width / 2)
    y_prob = y_distance / ((end_points[3] - end_points[1]) / 2)
    return max(x_prob, y_prob)


def contouring(thresh, mid, img, end_points, img_width, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy, img_width)
        return pos
    except:
        pass


def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def print_eye_pos(img, left, right, sunglasses_detected, threshold):
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_str = f'Left Eye Probability: {left:.2f}' if left is not None else 'Left Eye Probability: N/A'
    right_str = f'Right Eye Probability: {right:.2f}' if right is not None else 'Right Eye Probability: N/A'
    sunglasses_detected_str = 'Yes' if sunglasses_detected else 'No'
    threshold_str = f'Threshold: {threshold}'
    cv2.putText(img, left_str, (30, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, right_str, (30, 50), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'Sunglasses Detected: {sunglasses_detected_str}', (30, 70), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, threshold_str, (30, 90), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def get_input():
    device = input("Are you using a computer or a phone? (computer/phone): ").lower()
    while device not in ['computer', 'phone']:
        print("Invalid input! Please enter 'computer' or 'phone'.")
        device = input("Are you using a computer or a phone? (computer/phone): ").lower()

    glasses = input("Are you wearing eyeglasses? (yes/no): ").lower()
    while glasses not in ['yes', 'no']:
        print("Invalid input! Please enter 'yes' or 'no'.")
        glasses = input("Are you wearing eyeglasses? (yes/no): ").lower()

    return device, glasses


def eye_tracking(device, glasses,threshold, face_model, landmark_model, cap,results):
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]

    if glasses == "yes":
        threshold = 71
    else:
        threshold = 24

    ret, img = cap.read()
    thresh = img.copy()

    kernel = np.ones((9, 9), np.uint8)

    sunglasses_frames = 0

    while True:
        ret, img = cap.read()
        img_width = img.shape[1]
        rects = find_faces(img, face_model)

        # Draw detected faces on the image
        draw_faces(img, rects)
        draw_marks(img, rects, 155)
        sunglasses_detected = glasses

        if len(rects) == 0:
            print_eye_pos(img, None, None, sunglasses_detected, threshold)
        else:
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                # Draw landmarks on the image
                draw_marks(img, shape)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)

                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

                sunglasses = sunglasses_cascade.detectMultiScale(eyes_gray)
                if len(sunglasses) > 0:
                    sunglasses_frames += 1
                    if sunglasses_frames >= 3:
                        sunglasses_detected = True
                else:
                    sunglasses_frames = 0

                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)

                try:
                    eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left, img_width)
                    eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, img_width, True)
                    print_eye_pos(img, eyeball_pos_left, eyeball_pos_right, sunglasses_detected, threshold)

                    # Print in the terminal
                    left_prob = 1 - eyeball_pos_left if eyeball_pos_left is not None else None
                    right_prob = 1 - eyeball_pos_right if eyeball_pos_right is not None else None
                    sunglasses_detected_str = "Yes" if sunglasses_detected else "No"
                    print(f"Left Eye Probability: {left_prob:.2f} | Right Eye Probability: {right_prob:.2f} | Sunglasses Detected: {sunglasses_detected_str} | Threshold: {threshold}")
                except:
                    pass

        cv2.imshow('eyes', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def main():
    device, glasses = get_input()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return
    eye_tracking(device, glasses, cap)
    cap.release()
    cv2.destroyAllWindows()
