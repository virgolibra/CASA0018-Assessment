# CASA0018-Assessment
This is the repo for casa0018 assessment

Rock Paper Scissors Gesture Recognition on Raspberry Pi

GitHub Repo: https://github.com/virgolibra/CASA0018-Assessment

Minghao ZHANG

Apr 2022



---

The project is a simple gesture recognition project to identify rock, paper and scissors in real time by using Raspberry Pi and the camera. The project is based on the experiments with deep learning, simple image processing and image classification. 



**Research Question**

How to recognise three different gestures (rock, paper and scissors) based on deep learning and image classification with Raspberry Pi and camera.



---

##### Data

The data source is from Tensorflow dataset, called rock_paper_scissors. The dataset contains 2892 images with resolution 300*300 in RGB colour space and three different labels. 

Dataset Link: https://www.tensorflow.org/datasets/catalog/rock_paper_scissors



The default category for source is 372 images for test and 2520 for train. In this project, I split it into three sets as below:

+ Train set: 2142 images
+ Valid set: 372 images
+ Test set: 378 images

The ratio is about 75%, 12.5%, 12.5%.



A series of transform are implemented to simulate the captured images in real world.

+ Resize 300x300 images to 150x150
+ Colour transform, including brightness, hue, saturation, contrast and inversion.
+ Image rotation, flip, zoom.
+ Random function is applied to ensure randomly transform.
+ Convert RGB colour space to grayscale.
+ Shuffling the data to ensure the model not to learn something from the order



##### Model

Convolutional Neural Network (CNN) provides feature maps from input features based on shared-weight architecture, which has multiple applications in image and video recognition, image classification and image segmentation. Therefore, the model is based on CNN architecture with four convolution layers and three output units, which indicate to three gestures.



##### Raspberry Pi

To implement on Raspberry Pi, a 5 megapixels 1080p webcam OV5647 sensor is used to capture. The model is trained and saved based on the colab notebook and then loaded on Raspberry Pi for real-time image classification. 



##### Future Works

Successfully implement the TensorFlow Lite model is priority. Also, some components such as LCD screen and LEDs can be connected to Pi GPIO pins to represent the identification of rock, paper and scissors. As a kind of gesture recognition, it is possible to implement the gesture recognition to control some devices. For example, the light can be turned on by showing a paper, and be turned off by a rock.



----

## Installing OpenCV on the Raspberry Pi

[Installation Tutorial](https://pimylifeup.com/raspberry-pi-opencv/)

#### Install Packages

Update current installed packages

```
sudo apt update
sudo apt upgrade
```

Contain the tools to compile OpenCV code

```
sudo apt install cmake build-essential pkg-config git
```

Add image and video formats support

```
sudo apt install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev

sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```

Interface for OpenCV

```
sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
```

Packages for OpenCV to run at a decent speed on the Raspberry Pi

```
sudo apt install libatlas-base-dev liblapacke-dev gfortran
```

install relate to the Hierarchical Data Format (HDF5) that OpenCV uses to manage data

```
sudo apt install libhdf5-dev libhdf5-103
```

support for Python on our Raspberry Pi

```
sudo apt install python3-dev python3-pip python3-numpy
```



-----

#### Preparing Pi for Compiling OpenCV

Modify swap file configuration

```
sudo nano /etc/dphys-swapfile

# CONF_SWAPSIZE=100 replace with
CONF_SWAPSIZE=2048
```

Restart the service

```
sudo systemctl restart dphys-swapfile
```

Retrieve the latest available version of OpenCV from [git repo](https://github.com/opencv/opencv)

```
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```



----

#### Compiling OpenCV on Pi

Create the directory

```
mkdir ~/opencv/build
cd ~/opencv/build
```

Using `cmake` to prepare OpenCV for compilation on Raspberry Pi

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D BUILD_EXAMPLES=OFF ..
```

Using argument `-j$(nproc)`to tell the compiler to run a compiler for each of the available processors, which could significantly speed up the compilation process

```
make -j$(nproc)
```

*Alternative Command if above command is failed*

```
make
```

After finishing compilation, install OpenCV

```
sudo make install
```

Regenerate the operating systems library link cache

```
sudo ldconfig
```



----

#### Cleaning up after Compilation

Edit the swap file configuration back to original size

```
sudo nano /etc/dphys-swapfile

# CONF_SWAPSIZE=2048 replace with
CONF_SWAPSIZE=100
```

Restart service

```
sudo systemctl restart dphys-swapfile
```



----

#### Testing OpenCV on Pi

Using Python3 [installation](https://pimylifeup.com/installing-python-on-linux/)

Launch into python terminal by running

```
python3
```

Check OpenCV installation by importing a module

```
import cv2
```

With the module imported, the version should be retrieved by

```
cv2.__version__
```

And the version like`'4.1.2'` should be appeared in the command line





----

#### Troubleshooting

1. `No module named 'cv2'` (Solved):

   ```
   pip3 install opencv-python
   sudo apt-get install libqt-test
   sudo apt-get install libatlas-base-dev
   sudo apt-get install libjasper-dev
   sudo apt-get install libqtgui4
   ```

   Error removed with python3

2. `RuntimeError: module compiled against API version 0xe but this version of numpy is 0xd` (Solved):

   ```
   pip install numpy --upgrade
   ```

   Upgrade numpy



Output

```
Python 2.7.16 (default, Oct 10 2019, 22:02:15) 
[GCC 8.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import cv2
>>> cv2.__version__
'4.5.5-dev'
>>> 

```



[END]
