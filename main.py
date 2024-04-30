import cv2
import pandas as pd
import os
from threading import Thread
from queue import Queue
from face_detector import get_face_detector, find_faces, count_persons
from face_landmarks import get_landmark_model
from head_pose_estimation import head_main
from h5 import eye_tracking

def eye_tracking_thread(device, glasses, threshold, face_model, landmark_model, cap, results_queue):
    try:
        eye_tracking(device, glasses, threshold, face_model, landmark_model, cap, results_queue)
    except Exception as e:
        print(f"An error occurred in eye tracking thread: {e}")
        results_queue.put((None, None, None, f"Error in eye tracking thread: {e}"))

def head_pose_estimation_thread(cap, results_queue):
    try:
        head_main(cap, results_queue)
    except Exception as e:
        print(f"An error occurred in head pose estimation thread: {e}")
        results_queue.put((None, None, None))


def process_video(video_file):
    device, glasses = "computer", "yes"

    if device == 'computer' and glasses == 'yes':
        threshold = 71
    elif device == 'computer' and glasses == 'no':
        threshold = 24
    elif device == 'phone' and glasses == 'yes':
        threshold = 60
    else:
        threshold = 15

    face_model = get_face_detector()
    landmark_model = get_landmark_model()

    # Create VideoCapture object for the video file
    cap = cv2.VideoCapture(video_file)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract video name from the file path
    video_name = os.path.basename(video_file)

    # Create a CSV file to store the results
    results_file = '2.csv'

    with open(results_file, 'w') as f:
        f.write('Left_Eye_Probability,Right_Eye_Probability,People_Count,Head_Pose_Probability\n')

    # Create a Queue to communicate results between threads
    results_queue = Queue()

    # Start threads for different tasks
    eye_tracking_thread_obj = Thread(target=eye_tracking_thread,
                                     args=(device, glasses, threshold, face_model, landmark_model, cap, results_queue))
    eye_tracking_thread_obj.start()

    head_pose_estimation_thread_obj = Thread(target=head_pose_estimation_thread, args=(cap, results_queue))
    head_pose_estimation_thread_obj.start()

    # Process frames and update results
    frame_number = 0
    while cap.isOpened() and frame_number < total_frames:
        ret, frame = cap.read()
        if ret:
            faces = find_faces(frame, face_model)
            num_persons = count_persons(faces)

            # Get the results from the Queue
            left_eye_prob, right_eye_prob, head_pose_prob, error_message = results_queue.get()

            if left_eye_prob is None:
                # An error occurred in the threads, break out of the loop
                print("Error: An error occurred in processing threads.")
                break

            # Print the values before writing to the CSV
            print(
                f"Frame {frame_number}: Left Eye Probability = {left_eye_prob:.2f}, Right Eye Probability = {right_eye_prob:.2f}, Head Pose Probability = {head_pose_prob:.2f}, Number of Persons = {num_persons}")

            # Write the values to the CSV file
            with open(results_file, 'a') as f:
                f.write(f"{left_eye_prob},{right_eye_prob},{num_persons},{head_pose_prob}\n")

            frame_number += 1

        else:
            print("Error: Failed to capture frame.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    video_file = 'Videos/14.avi'  # Replace 'path_to_your_video.avi' with the actual path to your video file
    print(f"Processing video: {video_file}")
    process_video(video_file)

if __name__ == "__main__":
    main()
