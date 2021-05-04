import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import load_data

from torch.optim import lr_scheduler
from torchvision import models


device = torch.device('cpu')


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis
        # numpy images are formatted H x W x C
        # but torch images are formatted C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}


# trains a CNN based on Resnet18 pretrained model
# adapted from lab 9
def train_cnn(num_epochs=20):
    dataset_sizes, dataloaders = load_data.get_dataloaders('data/', 'data/labels/', transform_list=[ToTensor()])
    class_names = load_data.get_classes()
    num_classes = len(class_names)
    
    # get pretrained resnet
    model = models.resnet18(pretrained=True)

    # add new layer with our desired number of output classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # to GPU
    model = model.to(device)

    # start training
    criterion = nn.CrossEntropyLoss()

    # all parameters are being optimized
    # todo test different hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    # decays LR by factor of 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    since = time.time()

    # keep track of best model and its accuracy so far
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('----------------')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i, batch in enumerate(dataloaders[phase]):
                inputs, labels = batch["image"], batch["label"]
                
                # move data to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # update learning rate with scheduler
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            # copy the model with best accuracy on validation set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'models/cnn.pth')
    return model