# Importing the modules
import os
import glob
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import numpy as np
import tensorflow as tf
import tensorflow.keras
from PIL import Image, ImageOps
import argparse
import io
import picamera
import cv2
import sys
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10

def run(num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:

    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (150, 150, 1)), # shape 1
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation = 'relu'),
            tf.keras.layers.Dense(3, activation = 'softmax')
        ])

        rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(
            optimizer=rmsprop_optimizer,
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        return model

    model = create_model()
    model.summary()
    
    # Restore the weights
    model.load_weights('./checkpoints/my_checkpoint')

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, img_raw = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        img_raw = cv2.flip(img_raw, 1)
#         img = cv2.flip(img, 1)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis = 0)
# 
#         images = np.vstack([x])
#         classes = model.predict(images, batch_size = 10)
#         print(classes)
        	# let's downscale the image using new  width and height
        
        img = img_raw
        reduce_width = 150
        reduce_height = 150
        reduce_points = (reduce_width, reduce_height)
        img = cv2.resize(img, reduce_points, interpolation= cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         x = image.img_to_array(image)
        
#         x = image_gray
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
# 
        images = np.vstack([x])
        classes = model.predict(images, batch_size = 10)
        print(classes)
        if classes[0, 0] > 0.9:
            print('rock')
            class_text = 'ROCK'
        elif classes[0, 1] > 0.9:
            print('paper')
            class_text = 'PAPER'
        elif classes[0, 2] > 0.9:
            print('scissors')
            class_text = 'SCISSORS'
        else:
            print('none')
            class_text = 'NONE'
            
        # Calculate the FPS
        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
          end_time = time.time()
          fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
          start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = ' + str(int(fps))
        text_location = (_LEFT_MARGIN, _ROW_SIZE)
        cv2.putText(img_raw, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
        
        
        
        class_location = (_LEFT_MARGIN+50, _ROW_SIZE+50)
        cv2.putText(img_raw, class_text, class_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE+4, _TEXT_COLOR, _FONT_THICKNESS+2)
#         cv2.imshow('reduce size', img)
    # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
#         cv2.imshow('image_classification', img)
        cv2.imshow('Raw', img_raw)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        default=480) #640 480
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        default=480)
    args = parser.parse_args()

    run(int(args.numThreads),
        bool(args.enableEdgeTPU), int(args.cameraId), args.frameWidth,
        args.frameHeight)


if __name__ == '__main__':
    main()