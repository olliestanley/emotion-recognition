## emotion-recognition
Recognise faces and classify their emotions based on images or videos. Implemented in Python 3 using several publicly available libraries.

This repository contains code for HOG and SIFT feature extraction, and a bag of visual words implementation. It also contains code for training multiple emotion recognition models, two via sklearn (MLP and SVM) and one (CNN) via PyTorch, the most effective being the CNN tuned from ResNet18. The emotion recognition models must be fed images of faces; a separate pretrained facial recognition model is used to identify faces within images or videos before applying the emotion recognition model(s).

The dataset used for training of the models is not publicly available but the code here could easily be generalised to any similar dataset by only modifying the file `load_data.py`. Note also that some code is hacky due to the time constraints imposed on this project's implementation.
