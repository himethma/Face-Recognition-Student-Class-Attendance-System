import cv2
import csv
import os
import numpy as np
from simple_facerec import SimpleFacerec
from tensorflow.keras.models import load_model
from pymongo import MongoClient
from datetime import datetime

# Initialize SimpleFacerec to encode and recognize faces
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Folder containing known faces (named images)

# Load your trained Keras model
model_path = "my_model (2).keras"
model = load_model(model_path)

# Define the class labels 
class_labels = ["Dushara", "Himethma", "Kavindi", "Minhaj", "Nazhan", "Thathsara", "Umesh"]

# Mapping of student names to their IDs
student_ids = {
    "Dushara": "ID006",
    "Himethma": "ID003",
    "Kavindi": "ID002",
    "Minhaj": "ID001",
    "Nazhan": "ID005",
    "Thathsara": "ID004",
    "Umesh": "ID008",
}

# Define the late threshold time (9:30 AM)
late_threshold = datetime.strptime("09:30:00", "%H:%M:%S").time()

# Connect to MongoDB
client = MongoClient('mongodb+srv://himethma:himethma123@iotcluster.vze4lf7.mongodb.net/?retryWrites=true&w=majority&appName=IoTcluster')  # Replace with your connection string if using MongoDB Atlas
db = client["Face_Recognition"]  # Database name
attendance_collection = db["Attendance data"]  # Collection name

# Define CSV file for attendance logging
csv_file = "attendance.csv"

# Initialize the CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Student ID", "Date", "Time", "Status"])

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Track already logged names
logged_names = set()

# Preprocessing functions
def preprocess_face(face):
    """
    Preprocess the face image to match the model's input requirements.
    """
    face = cv2.resize(face, (180, 180))  # Resize to match model's input size
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    for i, face_loc in enumerate(face_locations):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        # Draw bounding box around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = "Unknown"  # Default label
        student_id = "Unknown"  # Default ID

        if face_names[i] != "Unknown":
            if face_names[i] in class_labels:  # Check if the recognized name is in class labels
                label = face_names[i]
                student_id = student_ids.get(face_names[i], "Unknown")  # Retrieve ID
            else:
                label = "Unknown"
        else:
            # Predict with the custom model if face is not recognized by SimpleFacerec
            face_roi = frame[y1:y2, x1:x2]
            try:
                processed_face = preprocess_face(face_roi)
                prediction = model.predict(processed_face)
                label_idx = np.argmax(prediction[0])  # Get the index of the highest probability
                confidence = prediction[0][label_idx]  # Get the confidence score
                
                if confidence > 0.8:  # Use the label only if confidence is high
                    label = class_labels[label_idx]
                    student_id = student_ids.get(label, "Unknown")
                else:
                    label = "Unknown"
                    student_id = "Unknown"
            except Exception as e:
                print(f"Error processing face: {e}")
                label = "Error"
                student_id = "Error"

        # Add label to the bounding box
        cv2.putText(frame, f"{label} ({student_id})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Log the name, ID, timestamp, and status if a face is recognized and not yet logged
        if label != "Unknown" and label != "Error" and label not in logged_names:
            logged_names.add(label)  # Add to the set of logged names
            current_datetime = datetime.now()
            current_date = current_datetime.strftime("%Y-%m-%d")
            current_time = current_datetime.strftime("%H:%M:%S")
            current_time_obj = current_datetime.time()
            
            # Determine attendance status based on the time
            status = "Late" if current_time_obj > late_threshold else "On Time"
            print(f"Recognized: {label} (ID: {student_id}) at {current_date} {current_time} - {status}")
            
            # Prepare the attendance record
            attendance_record = {
                "name": label,
                "student_id": student_id,
                "date": current_date,
                "time": current_time,
                "status": status
            }

            # Check if record already exists for this student and date
            existing_record = attendance_collection.find_one({
                "student_id": student_id,
                "date": current_date
            })

            if not existing_record:
                # Insert attendance record if it doesn't exist
                attendance_collection.insert_one(attendance_record)
                print(f"Attendance logged in MongoDB: {attendance_record}")
                
                # Append attendance to CSV file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([label, student_id, current_date, current_time, status])
                    print(f"Attendance logged in CSV: {label}, {student_id}, {current_date}, {current_time}, {status}")
            else:
                print(f"Attendance for {label} on {current_date} is already logged.")

    # Display the frame with bounding boxes and labels
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
