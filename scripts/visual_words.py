import numpy as np

from sklearn.cluster import MiniBatchKMeans


# apply the bag of visual words algorithm to extracted descriptors
# adapted from lab 7
def bag_visual_words(descriptor_list, y_train):
    descriptors = np.vstack(descriptor_list)

    # apply KMeans
    # we use k = number of classes * 10
    k = len(np.unique(y_train)) * 10

    # as per lab 7, MiniBatchKMeans is faster & lower memory usage
    batch_size = descriptors.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(descriptors)

    # descriptors -> histograms of codewords per image
    histograms = []
    indices = []

    for descriptor in descriptor_list:
        histogram = np.zeros(k)

        # predict (index of) class for desciptor
        idx = kmeans.predict(descriptor)
        indices.append(idx)

        # compute histogram
        for j in idx:
            histogram[j] = histogram[j] + (1 / len(descriptor))

        # append computed histogram to list
        histograms.append(histogram)

    # array easier to manipulate later
    hist_array = np.vstack(histograms)
    return kmeans, hist_array