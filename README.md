# Face-Recognition-with-OpenCV

This is a simple Face Recognition project using Python OpenCV, to learn and understand the basics of a project. This project is heavily derived and inspired by https://github.com/leodlca/lbph-face-recognition.

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
