import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3

# Initialize MediaPipe pose class and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def chair_stand_test():
    cap = cv2.VideoCapture(0)
    standing = False
    sit_to_stand_count = 0
    start_time = time.time()
    duration = 30  # Duration of the test in seconds

    stframe = st.empty()
    stcount = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to extract pose landmarks
        result = pose.process(rgb_frame)
        
        # Check if any landmarks are detected
        if result.pose_landmarks:
            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract relevant landmarks
            landmarks = result.pose_landmarks.landmark
            
            # Get coordinates for left and right hip, knee, and ankle
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles at the knees
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Determine if the person is standing or sitting
            if left_knee_angle > 160 and right_knee_angle > 160:
                current_position = "standing"
            elif left_knee_angle < 100 and right_knee_angle < 100:
                current_position = "sitting"
            else:
                current_position = "transition"
            
            # Detect transitions between sitting and standing
            if current_position == "standing" and not standing:
                standing = True
                sit_to_stand_count += 1
            elif current_position == "sitting":
                standing = False
        
        # Display the current frame
        stframe.image(frame, channels="BGR")
        stcount.write(f'Chair Stands: {sit_to_stand_count}')
        
        # Break the loop after 30 seconds
        if time.time() - start_time > duration:
            break
        
    cap.release()
    st.success(f'Test completed! Total chair stands: {sit_to_stand_count}')

def arm_curl_test():
    cap = cv2.VideoCapture(0)
    count = 0
    stage = None
    start_time = None

    # Setup Pose instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        stframe = st.empty()
        stcount = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Start timer on first frame
            if start_time is None:
                start_time = time.time()

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 30 - int(elapsed_time))

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    count += 1

                # Display count and remaining time
                cv2.putText(image, f'Count: {count}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'Angle: {round(angle, 2)}', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'Time: {remaining_time} sec', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            stframe.image(image, channels="BGR")
            stcount.write(f'Arm Curls: {count}')

            # Break the loop after 30 seconds
            if elapsed_time > 30:
                break

        cap.release()
        st.success(f'Test completed! Total arm curls: {count}')

# Streamlit app layouts
st.title('Senior Citizen Fitness Tests')
st.subheader('This app provides two fitness tests for senior citizens:')
#sidebar

#main
image1 = 'images/img1.jpg'
image2 = 'images/img2.png'

col1, col2 = st.columns(2)
with col1:
    st.image(image1, caption='Arm Curl Test', use_column_width=True)

with col2:
    st.image(image2, caption='Chair Stand Test', use_column_width=True)

st.write('1. Arm Curl Test: Counts the maximum number of curls performed in 30 sec.')
st.write('2. Chair Stand Test: Counts the maximum number of stand ups performed in 30 sec.')

test_option = st.selectbox('Select a test to perform', ['Arm Curl Test', 'Chair Stand Test'])

if st.button('Start Test'):
    if test_option == 'Arm Curl Test':
        st.write('Grab a chair and get ready for arm curls! You have 30 seconds.')
        time.sleep(3)
        arm_curl_test()
    elif test_option == 'Chair Stand Test':
        chair_stand_test()