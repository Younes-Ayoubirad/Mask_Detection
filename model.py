import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow.keras as tk
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import cv2

INIT_LR = 1e-4
EPOCHS = 3
BS = 32

image_paths = list(paths.list_images('data'))

images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = tk.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tk.preprocessing.image.img_to_array(image)
    image = tk.applications.mobilenet_v2.preprocess_input(image)
    images.append(image)
    labels.append(label)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tk.utils.to_categorical(labels)

x_train, x_test, y_train, y_test = (train_test_split(images, labels, test_size=0.2, random_state=42))

aug = tk.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

baseModel = tk.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

headModel = baseModel.output
headModel = tk.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = tk.layers.Flatten(name='flatten')(headModel)
headModel = tk.layers.Dense(128, activation="relu")(headModel)
headModel = tk.layers.Dropout(0.5)(headModel)
headModel = tk.layers.Dense(2, activation="softmax")(headModel)

model = tk.models.Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

lr_schedule = tk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)
opt = tk.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

H = model.fit(
    aug.flow(x_train, y_train, batch_size=BS),
    steps_per_epoch=len(x_train) // BS,
    validation_data=(x_test, y_test),
    validation_steps=len(x_test) // BS,
    epochs=EPOCHS,
)

model.save('mask_Detection.h5')




print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(['face_detector', 'deploy.prototxt'])
weightsPath = os.path.sep.join(['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNet(prototxtPath, weightsPath)
model = tk.model.load_model('mask_Detection.h5')

image = cv2.imread('./dataset/with_mask/0999.png')
orig = image.copy()
(h, w) = image.shape[:2]
# making standard image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

print("[INFO] computing face detections...")
# assign picture to openCv net
net.setInput(blob)
# predict all faces
detections = net.forward()


# Iterate over the detections
for i in range(0, detections.shape[2]):
    # Extract the confidence (i.e., probability) associated with the detection
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the confidence is greater than a threshold
    if confidence > 0.6:
        # Compute the (x, y)-coordinates of the bounding box for the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensure the bounding box does not fall outside the image dimensions
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Extract the face ROI, convert it from BGR to RGB, and preprocess it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = tk.preprocessing.image.img_to_array(face)
        face = tk.applications.mobilenet_v2.preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Pass the face through the model to determine if there is a mask or not
        (mask, withoutMask) = model.predict(face)[0]

        # Determine the class label and color we'll use to draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
