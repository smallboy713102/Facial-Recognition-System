# Facial-Recognition-System
The provided code is a Python script that implements a facial recognition attendance system using the face_recognition library and OpenCV. Here's a brief description of the code:

1. The code starts by importing the necessary libraries, including face_recognition, cv2 (OpenCV), csv, numpy, os, datetime, and gdown (for downloading files from Google Drive).

2. The Google Drive link for the training images folder is specified, along with a local path to store the downloaded images.

3. The code creates the training images folder if it doesn't already exist and downloads the images from the Google Drive link using the gdown library.

4. Known faces are loaded by iterating over the images in the training images folder. Each image is loaded using face_recognition.load_image_file, and the face encoding is obtained using face_recognition.face_encodings. The face encoding and corresponding name are added to the known_face_encodings and known_face_names lists, respectively.

5. The current date is obtained using the datetime library to create a unique CSV file for storing the attendance records.

6. The main loop of the code captures frames from the video source (usually a webcam) using cv2.VideoCapture.

7. Each captured frame is resized to a smaller size for faster face recognition processing.

8. Face locations and encodings are computed using the face_recognition library for the resized frame.

9. For each face detected, the code compares the face encoding with the known face encodings using face_recognition.compare_faces. If a match is found, the corresponding name is retrieved from the known_face_names list.

10. If the recognized name is present in the known_face_names list and hasn't been processed before, the code adds the name and current date to the CSV file as an attendance record. The processed names are stored in the processed_faces set to avoid duplicate entries.

11. The frame is annotated with the recognized name if the person is present, and the annotated frame is displayed in a window using cv2.imshow.

12. The loop continues until the 'q' key is pressed, upon which the video capture is released, windows are closed, and the CSV file is closed.

Overall, the code performs facial recognition on live video frames, matches the recognized faces with known faces, and maintains an attendance record by storing the names and dates in a CSV file.
