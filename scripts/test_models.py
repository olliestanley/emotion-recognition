import load_data
import os
import matplotlib.pyplot as plt
import load_data
import train_cnn
import random


# tests the model on the given video
# OpenCV's haar cascades detector will be used to identify faces
# then the CNN emotion recognition model will be used to recognise emotion
# todo implement choice of model here
def test_video(path_to_video):
    import torch
    import torch.nn as nn
    import random
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    from torchvision import models
    from matplotlib import rc
    import matplotlib
    from skimage.measure import label, regionprops
    from skimage import io, color, img_as_ubyte, img_as_float32, transform
    
    matplotlib.rcParams['animation.embed_limit'] = 2**128
    
    # test on the CPU only
    device = torch.device('cpu')
    
    # pytorch model; get dataloaders
    # ToTensor() transform is applied for PyTorch
    class_names = load_data.get_classes()
    num_classes = len(class_names)
    
    # define the model architecture
    model = models.resnet18(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)
    
    # load the saved state dictionary for the CNN model, on the CPU
    model.load_state_dict(torch.load('models/cnn.pth', map_location=device))
    model.eval()
    
    # load the cascade classifier used for face recognition
    # adapted from lab 6
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # load the video into memory using cv2 (code adapted from lab 5)
    cap = cv2.VideoCapture(path_to_video)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frameCount and ret:
        ret, video[fc] = cap.read()
        # convert from BGR (from OpenCV) to RGB colour
        video[fc] = cv2.cvtColor(video[fc], cv2.COLOR_BGR2RGB)
        fc += 1
    cap.release()
    
    rc('animation', html='jshtml')
    fig, ax = plt.subplots()

    def frame(i):
        ax.clear()
        ax.axis('off')
        fig.tight_layout()
        img = video[i, :, :, :]
        
        # convert for use with opencv
        img_gray = img_as_ubyte(color.rgb2gray(img))
        
        # detect faces using cascade
        faces = face_classifier.detectMultiScale(img_gray, 1.5, 3)
        
        for face in faces:
            x_min = face[0]
            x_max = x_min + face[3]
            y_min = face[1]
            y_max = y_min + face[2]
            
            # draw box around face
            ax.add_patch(patches.Rectangle(xy=(face[0], face[1]), width=face[2], height=face[3], fill=False, color='r'))
            
            # img shape is 720x1080x3
            # we need to extract face and convert it to 3x100x100 then into a tensor
            
            # extract the face region
            face_img = img[y_min:y_max, x_min:x_max, :]
            # convert to float32 for the CNN
            face_img = img_as_float32(face_img)
            # resize to 100x100 for CNN
            face_img = transform.resize(face_img, (100, 100))
            # transpose to torch format
            face_img = face_img.transpose((2, 0, 1))
            # convert from numpy array to torch tensor
            face_img = torch.from_numpy(face_img)
            # add additional dimension (3x100x100 -> 1x3x100x100)
            face_img = face_img.unsqueeze(0)
            
            # pass face to cnn
            output = model(face_img)
            _, pred = torch.max(output, 1)
            label = class_names[pred[0]]
            
            # write CNN output label on image
            ax.text(x_min, y_min - 3, label, color='green', zorder=100000)
            
        plot = ax.imshow(img)
        return plot

    anim = animation.FuncAnimation(fig, frame, frames=frameCount)
    plt.close()
    return anim


# tests the given model on the given test set
# model_type can be 'cnn', 'hog_mlp', or 'sift_svm'
def test_model(path_to_testset, model_type):
    if model_type == 'cnn':
        import torch
        import torch.nn as nn
        import random
        
        from torchvision import models
        
        # test on the CPU only
        device = torch.device('cpu')
        
        # pytorch model; get dataloaders
        # ToTensor() transform is applied for PyTorch
        dataset_sizes, dataloaders = load_data.get_dataloaders(path_to_testset, os.path.join(path_to_testset, 'labels/'), transform_list=[train_cnn.ToTensor()], sets=['test'])
        class_names = load_data.get_classes()
        num_classes = len(class_names)
        
        # define the model architecture
        model = models.resnet18(pretrained=True)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes)
        
        # load the saved state dictionary for the CNN model, on the CPU
        model.load_state_dict(torch.load('models/cnn.pth', map_location=device))
        model.eval()
        
        # function which transforms a tensor format image to a numpy one and displays it
        def imshow(axe, img, title):
            img = img.numpy().transpose((1, 2, 0))
            axe.imshow(img)
            axe.set_title(title)
            axe.set_axis_off()
        
        # track correct classifications so far
        running_corrects = 0
        # track total displayed images so far (we want to display 4 total)
        displayed = 0
        
        # define axes for displaying images later
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
        ax = axes.ravel()
        
        for i, batch in enumerate(dataloaders['test']):
            # get images and labels for this batch
            inputs, labels = batch["image"], batch["label"]
            
            # ensure images and labels are on the CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # apply model to images
            outputs = model(inputs)
            # outputs -> class predictions
            _, preds = torch.max(outputs, 1)
            # add number of correct predictions to sum
            running_corrects += torch.sum(preds == labels.data)
            
            if displayed < 4:
                axe = ax[displayed]
                # select one random image in batch (batch size is 4)
                j = random.randint(0, 3)
                imshow(axe, inputs.data[j], f"Label: {class_names[labels[j]]} \n Prediction: {class_names[preds[j]]}")
                displayed += 1
        
        # display the example images with tight layout
        fig.tight_layout()
        plt.show()
        
        # number of correct predictions / total number of test images
        test_acc = running_corrects.double() / dataset_sizes['test']
        print('Test set accuracy:', test_acc)
    else:
        import pickle
        import extract_descriptors

        from sklearn import metrics

        # sklearn model; get lists
        test_images, test_labels = load_data.get_test_lists(path_to_testset, os.path.join(path_to_testset, '/labels/'))
        class_names = load_data.get_classes()
        
        if model_type == 'hog_mlp':
            # load the saved sklearn MLP classifier
            classifier = pickle.load(open('models/hog_mlp.model', mode='rb'))
            
            # extract feature descriptors for each test set image using HOG
            descriptors, y_test = extract_descriptors.extract_descriptors(test_images, test_labels, 'hog')
            # predict labels based on test set descriptors
            predicted = classifier.predict(descriptors)
            
            # display 4 images from the test set with predictions and true labels
            fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
            ax = axes.ravel()
            for i in range(4):
                ax[i].imshow(test_images[i])
                ax[i].set_title(f'Label: {class_names[y_test[i]]} \n Prediction: {class_names[predicted[i]]}')
                ax[i].set_axis_off()
            fig.tight_layout()
            plt.show()
            
            # output sklearn classification report
            print(f"""Classification report for classifier {classifier}:\n {metrics.classification_report(y_test, predicted)}""")
        elif model_type == 'sift_svm':
            import numpy as np
            
            # load the saved SVM classifier
            classifier = pickle.load(open('models/sift_svm.model', mode='rb'))
            # also load the kmeans model used for the visual bag of words implementation
            kmeans = pickle.load(open('models/vbow_kmeans.model', mode='rb'))
            
            # extract feature descriptors for each test image using SIFT
            descriptors, y_test = extract_descriptors.extract_descriptors(test_images, test_labels, 'sift')
            
            # next build histograms with visual bag of words for the test images
            k = len(np.unique(y_test)) * 10
            test_histograms = []
            for i in range(len(descriptors)):
                desc = descriptors[i]
                if desc is not None:
                    histogram = np.zeros(k)
                    for j in kmeans.predict(desc):
                        histogram[j] = histogram[j] + (1 / len(desc))
                    test_histograms.append(histogram)
                else:
                    test_histograms.append(None)
            
            # remove images without any descriptors
            idx_not_empty = [i for i, x in enumerate(test_histograms) if x is not None]
            test_hist_array = np.vstack([test_histograms[i] for i in idx_not_empty])
            y_test = [y_test[i] for i in idx_not_empty]

            # apply the SVM classifier to the histograms
            predicted = classifier.predict(test_hist_array).tolist()

            # display 4 images with predictions and true labels from the test set
            fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
            ax = axes.ravel()
            for i in range(4):
                ax[i].imshow(test_images[i])
                ax[i].set_title(f'Label: {class_names[y_test[i]]} \n Prediction: {class_names[predicted[i]]}')
                ax[i].set_axis_off()
            fig.tight_layout()
            plt.show()

            # output sklearn classification report for the SVM
            print(f"""Classification report for classifier {classifier}: {metrics.classification_report(y_test, predicted)}\n""")
        else:
            print('Invalid model type specified, valid options are [cnn, hog_mlp, sift_svm]')