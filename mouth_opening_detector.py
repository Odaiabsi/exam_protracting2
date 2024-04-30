import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import numpy as np

def record_mouth_distances(face_model, landmark_model, cap, outer_points, d_outer, inner_points, d_inner, record_duration=100):
    frame_count = 0
    while frame_count < record_duration:
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            shape = detect_marks(img, landmark_model, rect)
            draw_marks(img, shape)
            cv2.putText(img, f'Recording Mouth distances: {frame_count}/{record_duration}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)
            cv2.imshow("Output", img)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for i, (p1, p2) in enumerate(outer_points):
            d_outer[i] += shape[p2][1] - shape[p1][1]
        for i, (p1, p2) in enumerate(inner_points):
            d_inner[i] += shape[p2][1] - shape[p1][1]

    cv2.destroyAllWindows()
    d_outer[:] = [x / record_duration for x in d_outer]
    d_inner[:] = [x / record_duration for x in d_inner]

def detect_mouth_open(face_model, landmark_model, cap, outer_points, d_outer, inner_points, d_inner):
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cap.read()
        rects = find_faces(img, face_model)
        for rect in rects:
            try:
                shape = detect_marks(img, landmark_model, rect)
                cnt_outer = 0
                cnt_inner = 0
                draw_marks(img, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    try:
                        if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                            cnt_outer += 1
                    except:
                        pass
                for i, (p1, p2) in enumerate(inner_points):
                    try:
                        if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                            cnt_inner += 1
                    except:
                        pass
                if cnt_outer > 3 and cnt_inner > 2:
                    print('Mouth open')
                    cv2.putText(img, 'Mouth open', (30, 30), font,
                            1, (0, 255, 255), 2)
            except:
                pass
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main(cap,results):
    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    # Initialize variables for mouth detection
    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = np.zeros(5, dtype=np.int64)
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = np.zeros(3, dtype=np.int64)

    # record_mouth_distances(face_model, landmark_model, cap, outer_points, d_outer, inner_points, d_inner)
    detect_mouth_open(face_model, landmark_model, cap, outer_points, d_outer, inner_points, d_inner)

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows


