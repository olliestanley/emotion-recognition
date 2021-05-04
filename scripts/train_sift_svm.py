import numpy as np
import load_data
import extract_descriptors
import visual_words
import pickle

from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from skimage import img_as_ubyte, color


# returns the trained SVM model and the KMeans model used for VBoW
# if vbow=False, returns only trained SVM model
def train_sift_svm(vbow=True, save_model=True):
    # load list of training images and labels, including the validation set
    train_images, train_labels = load_data.get_train_lists('data/', 'data/labels/', include_val=True)
    # extract feature descriptors for each image using SIFT
    descriptors, y_train = extract_descriptors.extract_descriptors(train_images, train_labels, 'sift')
    
    if vbow:
        # apply the KMeans Bag of Visual Words approach to obtain histograms for training
        kmeans, hist_array = visual_words.bag_visual_words(descriptors, y_train)
    
    # parameters to test in gridsearch
    parameters = {
        # which kernel to use for the SVM
        'kernel': ('linear', 'rbf', 'poly'),
        # regularisation strength determinant
        'C': [1, 5, 10],
        # polynomial degree if poly kernel used
        'degree': [3, 4, 5]
    }
    
    # then fit an SVM on histograms as x
    classifier = svm.SVC()
    # use RandomizedSearch with 5-fold crossvalidation to tune hyperparameters
    classifier = RandomizedSearchCV(classifier, parameters)
    if vbow:
        # fit using BoVW histograms as x, labels as y
        classifier.fit(hist_array, y_train)
    else:
        # fit using SIFT descriptors as x, labels as y
        classifier.fit(np.vstack(descriptors), y_train)
    
    # output optimal parameters found by the search
    print('Optimal:\n', classifier.best_params_)
    
    # save tuned model
    if save_model:
        pickle.dump(classifier, open('models/sift_svm.model', mode='wb'))
        if vbow:
            pickle.dump(kmeans, open('models/vbow_kmeans.model', mode='wb'))
    
    if vbow:
        return classifier, kmeans
    else:
        return classifier