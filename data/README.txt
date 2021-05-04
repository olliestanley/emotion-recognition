The data used in the original run of this project is not publicly available. However, any dataset of human faces could be used here.

Data must be split into 'train', 'test', and 'val' folders, each containing a subset of images. For best results ensure images are all of identical dimension and contain only a single face.

Additionally, a fourth folder called 'labels' must be included. This must contain a separate txt file for each subset, named 'list_label_train.txt', 'list_label_val.txt', and 'list_label_test.txt'. Each must contain a line-separated list of filename-label combinations (separated by a single space). Labels must range from 1 to 7.

In order, labels 1 through 7 represent 7 emotions: 'Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral'.