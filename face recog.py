import face_recognition
import cv2
import csv
import numpy as np
import os
from datetime import datetime
#import gdown

# Google Drive link to the training images folder
#training_images_drive_link = "https://drive.google.com/drive/folders/XXXXXXXXXXXXX"
# Create the training images folder if it doesn't exist
#if not os.path.exists(training_images_folder):
    #os.makedirs(training_images_folder)

# Download the training images from Google Drive
#gdown.download(training_images_drive_link, output=training_images_folder, quiet=False)

# Path to the input images folder
input_images_folder = "input_images"

# Path to the training images folder
training_images_folder = "training_images"

# Get the list of input image files
input_image_files = os.listdir(input_images_folder)

# Load known faces
known_face_encodings = []
known_face_names = []

# Iterate over the training images folder
for file in os.listdir(training_images_folder):
    if file.endswith(".jpg") or file.endswith(".png"):  # Skip non-image files
        image_path = os.path.join(training_images_folder, file)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(file)[0])

# Get date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file
csv_file_path = f"{current_date}.csv"
with open(csv_file_path, "w+", newline="") as f:
    lnwriter = csv.writer(f)

    # Iterate over input images
    for input_file in input_image_files:
        input_image_path = os.path.join(input_images_folder, input_file)
        input_image = cv2.imread(input_image_path)

        # Check if the input image was loaded successfully
        if input_image is None:
            print(f"Failed to load image: {input_image_path}")
            continue

        face_locations = face_recognition.face_locations(input_image)

        # Check if any faces were detected
        if len(face_locations) == 0:
            print(f"No faces found in image: {input_image_path}")
            continue

        face_encodings = face_recognition.face_encodings(input_image, face_locations)

        processed_faces = set()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Add the text if person is present and not already processed
                if name in known_face_names and name not in processed_faces:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCorner = (face_location[3], face_location[2] + 30)
                    fontsScale = 1
                    fontColor = (255, 255, 0)
                    thick = 2
                    linetype = 2
                    cv2.putText(input_image, name, bottomLeftCorner, font,
                                fontsScale, fontColor, thick, linetype)
                    lnwriter.writerow([name, current_date])
                    processed_faces.add(name)

        # Display the image with predictions
        cv2.imshow("Attendance", input_image)
        cv2.waitKey(0)  # Delay between each image display

# Destroy any remaining OpenCV windows
cv2.destroyAllWindows()





















