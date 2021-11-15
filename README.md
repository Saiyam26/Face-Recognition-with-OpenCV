# Face-Recognition-with-OpenCV

- This is a simple Face Recognition project using Python OpenCV, to learn and understand the basics of a project. This project is inspired by https://github.com/leodlca/lbph-face-recognition.
- For Face Detection we use OpenCV's cascade: *haarcascade_frontalface_alt.xml*, which can be found at https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml.
- It uses the LPBH Algorithm for training and recognizing the different faces.

### Requirements

- Python 3.6+
- OpenCV
- Numpy
- Pandas

### Taking Photos
1. Run `python TakePhotos.py`
2. Enter an ID and Name
3. Press the `s` key repeatedly to take photos, once a box appears around your face. It is recommended to take atleast 25 pictures.
4. Press the `q` key when you're finished taking pictures.

### Training the Model
1. Run `python Train.py`
2. After Training is complete the program will generate the file "Classifiers/TrainedLBPH.yml"

### Recognizing
1. Run `python Recognize.py`
