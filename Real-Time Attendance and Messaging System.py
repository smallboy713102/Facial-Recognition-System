import face_recognition
import cv2
import csv
import numpy as np
import os
from datetime import datetime
from twilio.rest import Client


account_sid = "your ACCOUNT SID"
auth_token = "your AUTHORISED TOKEN"
twilio_phone_number = "your twilio PHONE NUMBER" 

# Google Drive link to the training images folder
# training_images_drive_link = "https://drive.google.com/drive/folders/XXXXXXXXXXXXX"

training_images_folder = "training_images"


if not os.path.exists(training_images_folder):
    os.makedirs(training_images_folder)

# Download the training images from Google Drive
# gdown.download(training_images_drive_link, output=training_images_folder, quiet=False)


known_face_encodings = []
known_face_names = []


for file in os.listdir(training_images_folder):
    if file.endswith(".jpg") or file.endswith(".png"):  
        image_path = os.path.join(training_images_folder, file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file)[0])

students = known_face_names.copy()

# Get date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file
csv_file_path = f"{current_date}.csv"
f = open(csv_file_path, "w+", newline="")
lnwriter = csv.writer(f)

processed_faces = set()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Function to send attendance message
def send_attendance_message(name):
    # Initialize the Twilio client
    client = Client(account_sid, auth_token)

    # Phone numbers to send the message to (replace with actual numbers)
    recipient_numbers = {
        "Name":"Phone number"
    }

    # Compose the attendance message
    message = f"Attendance recorded for {name}. This is a Facial Recognition System and Attendance System made by " \
              f"Nilesh Ranjan Pal, KGEC'25"

    # Send the attendance message to each recipient
    message = client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=recipient_numbers[name]
    )

    print(f"Message sent to {name} with message ID: {message.sid}")

    print("Attendance messages sent to all recipients.")

while True:
    # Read a single frame from the video capture
    ret, frame = video_capture.read()

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if person is present and not already processed
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_corner = (face_location[3] * 4, face_location[2] * 4 + 30)
                font_scale = 1
                font_color = (255, 0, 0)
                thickness = 2
                line_type = 2
                cv2.putText(frame, name + " is present", bottom_left_corner, font,
                            font_scale, font_color, thickness, line_type)

            if name in known_face_names and name not in processed_faces:
                lnwriter.writerow([name, current_date])
                processed_faces.add(name)
                send_attendance_message(name)  # Send attendance message to the mobile device

    # Display the resulting frame
    cv2.imshow("Attendance", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
f.close()
