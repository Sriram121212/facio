from flask import Flask
import tkinter as tk
import dlib
import numpy as np
import cv2
import pandas as pd
import os
import logging
import requests
from datetime import datetime, time,timezone
from tkinter import messagebox
import csv


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/sakthi')
def sakthi():
    return 'Hello, sakthi!'


@app.route("/dhoni")
def dhoni():
    window=tk.Tk()
    window.title("Face recognition system")
    def Training():
        # Path of cropped faces
        path_images_from_camera = "data/data_faces_from_camera/"

        # Use Dlib's frontal face detector
        detector = dlib.get_frontal_face_detector()

        # Get face landmarks
        predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

        # Use Dlib's ResNet50 model for 128D face descriptor
        face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


        def return_128d_features(path_img):
            """Extracts 128D facial features from an image."""
            img_rd = cv2.imread(path_img)
            faces = detector(img_rd, 1)

            if len(faces) != 0:
                shape = predictor(img_rd, faces[0])
                face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
            else:
                face_descriptor = 0  # No face detected
            return face_descriptor


        def return_features_mean_personX(path_face_personX):
            """Computes the average (mean) of 128D face descriptors for a person."""
            features_list_personX = []
            photos_list = os.listdir(path_face_personX)

            for photo in photos_list:
                features_128d = return_128d_features(os.path.join(path_face_personX, photo))
                if features_128d != 0:
                    features_list_personX.append(features_128d)

            if features_list_personX:
                return np.array(features_list_personX, dtype=object).mean(axis=0)
            else:
                return np.zeros(128, dtype=object, order='C')


        def get_existing_employees(csv_path):
            """Reads existing employees from the CSV file to avoid duplicates."""
            if not os.path.exists(csv_path):
                return set()  # If file doesn't exist, return an empty set

            existing_names = set()
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row:
                        existing_names.add(row[0])  # First column is the employee name
            return existing_names


        def main():
            logging.basicConfig(level=logging.INFO)
            csv_path = "data/features_all.csv"

            # Get list of existing employees
            existing_names = get_existing_employees(csv_path)

            # Get all persons from dataset
            person_list = sorted(os.listdir(path_images_from_camera))

            with open(csv_path, "a", newline="") as csvfile:  # Open in append mode
                writer = csv.writer(csvfile)

                for person in person_list:
                    # Extract employee name from folder
                    if len(person.split('_', 2)) == 2:
                        person_name = person  # "person_x"
                    else:
                        person_name = person.split('_', 2)[-1]  # "person_x_tom"

                    # Skip if employee already exists
                    if person_name in existing_names:
                        logging.info(f"Skipping {person_name} (already in CSV)")
                        continue

                    logging.info(f"Processing new employee: {person_name}")

                    features_mean_personX = return_features_mean_personX(os.path.join(path_images_from_camera, person))
                    features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)  # Insert name at start

                    writer.writerow(features_mean_personX)
                    logging.info(f"Added new employee: {person_name}")

            logging.info(f"Updated features saved in {csv_path}")


        if __name__ == '__main__':
            main()
        messagebox.showinfo('Result','Training dataset completed!!!')

    b1=tk.Button(window,text="TRAINING",font=("Times New Roman",20),bg='green',fg='white',command=Training)
    b1.grid(column=1, row=4)

    
    window.geometry("800x200")
    window.mainloop()
    return "Face Trining completed!" 
@app.route("/detect")
def detect():
    window=tk.Tk()
    window.title("Face recognition system")
    def Detection():

        # Load DNN model for face detection
        face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )

        # Load Dlib models
        predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

        class FaceRecognizer:
            def __init__(self):
                self.face_feature_known_list = []
                self.face_name_known_list = []
                self.frame_skip = 3
                self.frame_cnt = 0

            def get_face_database(self):
                if os.path.exists("data/features_all.csv"):
                    csv_rd = pd.read_csv("data/features_all.csv", header=None)
                    self.face_name_known_list = csv_rd.iloc[:, 0].tolist()
                    self.face_feature_known_list = csv_rd.iloc[:, 1:].values
                    logging.info(f"Loaded {len(self.face_name_known_list)} faces from database")
                    return True
                else:
                    logging.warning("Face database not found!")
                    return False

            @staticmethod
            def return_euclidean_distance(feature_1, feature_2):
                feature_1 = np.array(feature_1)
                feature_2 = np.array(feature_2)
                return np.linalg.norm(feature_1 - feature_2)

            def save_attendance_to_api(self, name_str, check_out=False):
                #current_datetime = datetime.now()
                #current_time_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                # Get the current time in UTC
                current_datetime = datetime.now(timezone.utc)
                # Format it as per the UTC ISO 8601 format
                current_time_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
                name, employee_id = name_str.split('_')

                if not check_out:
                    # For check-in
                    payload = {
                        "employee_ID": employee_id,
                        "name": name,
                        "attendance": {
                            "checkIn": current_time_str,
                            #"checkOut": None,
                            #"notes": "Arrived on time"
                        }
                    }
                    url = "https://hrm.quantumsharq.com/api/attendance/check-in"
                    # Send POST request for check-in
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        logging.info(f"Check-in recorded for {name} at {current_time_str}.")
                    else:
                        logging.error(f"Failed to record check-in for {name}: {response.text}")
                else:
                    # For check-out
                    payload = {
                        "employee_ID": employee_id,
                        "checkOut": current_time_str,
                        "notes": "Completed all tasks"
                        }
                    
                    url = "https://hrm.quantumsharq.com/api/attendance/check-out"
                    # Send PATCH request for check-out
                    response = requests.post(url, json=payload)
                    if response.status_code == 200:
                        logging.info(f"Check-out recorded for {name} at {current_time_str}.")
                    else:
                        logging.error(f"Failed to record check-out for {name}: {response.text}")

            def is_check_in_time(self):
                current_time = datetime.now().time()
                start_time = time(9, 25)  # 8:00 AM
                end_time = time(10, 0)   # 12:00 PM
                return start_time <= current_time <= end_time

            def is_check_out_time(self):
                current_time = datetime.now().time()
                check_out_time = time(18, 30)  # 5:00 PM
                return current_time >= check_out_time

            def detect_faces(self, img):
                blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                h, w = img.shape[:2]
                faces = []
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.6:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        faces.append((startX, startY, endX, endY))
                return faces

            def recognize_faces(self, img, faces):
                face_names = []
                for (startX, startY, endX, endY) in faces:
                    shape = predictor(img, dlib.rectangle(startX, startY, endX, endY))
                    face_descriptor = np.array(face_reco_model.compute_face_descriptor(img, shape))
                    
                    # Compare with known faces
                    distances = [self.return_euclidean_distance(face_descriptor, known_face) for known_face in self.face_feature_known_list]
                    min_distance = min(distances)
                    name = "Unknown"
                    if min_distance < 0.4:
                        name = self.face_name_known_list[distances.index(min_distance)]
                    
                    face_names.append(name)

                    # Draw rectangle and label
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 1)
                    cv2.putText(img, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Save attendance only if the person is known and during the check-in or check-out time
                    if name != "Unknown":
                        if self.is_check_in_time():
                            self.save_attendance_to_api(name)
                        elif self.is_check_out_time():
                            self.save_attendance_to_api(name, check_out=True)

                return img

            def process(self, stream):
                if not self.get_face_database():
                    return

                while stream.isOpened():
                    self.frame_cnt += 1
                    ret, frame = stream.read()

                    if not ret:
                        break

                    # Process every nth frame
                    if self.frame_cnt % self.frame_skip == 0:
                        # Resize frame for faster processing
                        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                        # Detect and recognize faces
                        faces = self.detect_faces(small_frame)
                        frame_with_names = self.recognize_faces(small_frame, faces)

                        # Show the result
                        cv2.imshow("Face Recognition", frame_with_names)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                stream.release()
                cv2.destroyAllWindows()

            def run(self):
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.process(cap)

        if __name__ == "__main__":
            logging.basicConfig(level=logging.INFO)
            face_recognizer = FaceRecognizer()
            face_recognizer.run()
    


    b1=tk.Button(window,text="DETECTION",font=("Times New Roman",20),bg='pink',fg='black',command=Detection)
    b1.grid(column=2, row=4)
        
    window.geometry("800x200")
    window.mainloop()
    return "Face detection completed!" 

if __name__ == '__main__':
    app.run(debug=True)


