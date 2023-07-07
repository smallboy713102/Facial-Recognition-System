import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

class_labels = ['Nilesh', 'Arghya', 'Devleena','Ankan','Debayan',
                "LordGD","Mayukh","Rizzu","Vaibhavi","Arindam",
                "Manab","Swangdipta","Udit","Ankesh"]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    resized_frame = cv2.resize(frame, (128, 128))

    preprocessed_frame = resized_frame.astype(np.float32) / 255.0

    input_data = np.expand_dims(preprocessed_frame, axis=0)

    prediction = model.predict(input_data)


    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index]


    label = f'{predicted_class_label}: {confidence*100:.2f}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    cv2.imshow('Image Classification', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
