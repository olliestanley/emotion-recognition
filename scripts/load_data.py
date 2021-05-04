import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from skimage import io, img_as_float32
from torchvision import transforms


# list of all class names in the dataset
class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']


# custom Dataset class specific to the face Dataset
class FaceDataset(Dataset):
    def __init__(self, images, images_dir, transform=None):
        # create list of image filenames
        self.image_files = [i for i in images.keys()]
        # store images directory for later
        self.images_dir = images_dir
        # labels 1-7 on file but subtract 1 from each -> 0-6
        self.labels = [int(v) - 1 for v in images.values()]
        # store transform for use when loading images
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get the filename for the image
        img_name = self.image_files[idx]
        # get the path to the image file
        img_path = os.path.join(self.images_dir, img_name)
        # load the image in float32 form
        img = img_as_float32(io.imread(img_path))
        # get the label for the image
        label = np.array(self.labels[idx]).astype(np.int64)
        # return image and label in a dict
        sample = {'image': img, 'label': label}
        
        # apply transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# get a dataloader corresponding to a dataset for each phase (default train, val, test)
def get_dataloaders(data_dir, labels_dir, sets=['train', 'val', 'test'], transform_list=None, batch_size=4):
    # for each phase load dict of filenames to labels
    lbls = {}
    for phase in sets:
        with open(os.path.join(labels_dir, 'list_label_' + phase + '.txt')) as f:
            lbls[phase] = {x.split(' ')[0] : x.split(' ')[1] for x in f}
    
    # create FaceDataset for each phase
    if transform_list:
        tform = transforms.Compose(transform_list)
    else:
        tform = None
    datasets = {x : FaceDataset(lbls[x], data_dir + x, tform) for x in sets}
    # get sizes of each dataset
    dataset_sizes = {x : len(datasets[x]) for x in sets}
    # create dataloaders for each phase's FaceDataset
    dataloaders = {k : DataLoader(v, batch_size=batch_size, shuffle=True, num_workers=0) for k,v in datasets.items()}
    
    # then return the sizes and loaders
    return dataset_sizes, dataloaders


# get lists of training images and labels
# include_val includes validation images and labels too
def get_train_lists(data_dir, labels_dir, transform_list=None, include_val=True):
    dataset_sizes, dataloaders = get_dataloaders('data/', 'data/labels/', sets=['train','val'])
    
    train_images = []
    train_labels = []
    
    # go through each training batch and append to the list
    for i, batch in enumerate(dataloaders['train']):
        inputs, labels = batch["image"], batch["label"]
    
        train_images.extend(inputs)
        train_labels.extend(labels)
    
    if include_val:
        # go through each validation batch and append to the list
        for i, batch in enumerate(dataloaders['val']):
            inputs, labels = batch["image"], batch["label"]
    
            train_images.extend(inputs)
            train_labels.extend(labels)
    
    return train_images, train_labels


# get lists of test set images and labels
def get_test_lists(data_dir, labels_dir, transform_list=None):
    dataset_sizes, dataloaders = get_dataloaders('data/', 'data/labels/', sets=['test'])
    
    test_images = []
    test_labels = []
    
    for i, batch in enumerate(dataloaders['test']):
        inputs, labels = batch["image"], batch["label"]
    
        test_images.extend(inputs)
        test_labels.extend(labels)
    
    return test_images, test_labels


# gets a list of class names
# list index of a class is equal to numerical value representing it in dataset
def get_classes():
    return class_names