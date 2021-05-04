import load_data
import extract_descriptors
import pickle
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier


# trains a hog_mlp model on the training set
def train_hog_mlp():
    train_images, train_labels = load_data.get_train_lists('data/', 'data/labels/')
    
    # extract hog descriptors from each training set image
    descriptor_list, y_train = extract_descriptors.extract_descriptors(train_images, train_labels, 'hog')
    # array easier to manipulate later
    descriptors = np.vstack(descriptor_list)
    
    # parameters to test in gridsearch
    parameters = {
        'learning_rate_init': [0.1, 0.01, 0.001],
        'solver': ['sgd', 'adam'],
        'hidden_layer_sizes': [(100), (50, 20), (50, 30, 10)],
        'alpha': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }

    
    # create MLP classifier
    classifier = MLPClassifier(max_iter=1000, random_state=1)
    # use RandomizedSearch with 5-fold crossvalidation to tune hyperparameters
    classifier = RandomizedSearchCV(classifier, parameters)
    # fit on HOG descriptors as x
    classifier.fit(descriptors, y_train)
    
    # output the optimal parameters found by the search
    print('Optimal:\n', classifier.best_params_)
    
    # save the MLP classifier to file and return it
    pickle.dump(classifier, open('models/hog_mlp.model', mode='wb'))
    return classifier