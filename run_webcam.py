# Teachable Machine & Python OpenCV

import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
# image = Image.open('img01.jpeg')
# img = cv2.imread('img01.jpeg')

# Webcam Video input
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.resize(img, (224, 224))

    #turn the image into a numpy array
    image_array = np.asarray(img)
    # print(image_array.shape)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array
    # print(data[0])

    # run the inference
    pred = model.predict(data)
    print(np.argmax(pred))
    threshold = round(np.max(pred), 2)

    i = np.argmax(pred)
    labels = ["Thumb-Up", "Thumb-Down", "None"]

    img = cv2.resize(img, (500, 500))
    cv2.putText(img, labels[i], (10, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
    cv2.putText(img, str(round(threshold * 100, 2)) + " %", (360, 30), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
    cv2.imshow('result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

