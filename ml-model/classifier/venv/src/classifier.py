import numpy as np
import pandas as pd
import seaborn as sb
import cv2
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session

TRAIN_DATASET_DIR_PATH = "/mnt/c/Users/Corbin/source/repos/Dog-Breed-Classification/ml-model/classifier/data/train"
TEST_DATASET_DIR_PATH = "/mnt/c/Users/Corbin/source/repos/Dog-Breed-Classification/ml-model/classifier/data/test"
LABELS_FILE_PATH = "/mnt/c/Users/Corbin/source/repos/Dog-Breed-Classification/ml-model/classifier/data/labels.csv"

def main():
    training_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testing_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    training_dataset = datasets.ImageFolder(TRAIN_DATASET_DIR_PATH, transform=training_transforms)
    testing_dataset = datasets.ImageFolder(TEST_DATASET_DIR_PATH, transform=testing_transforms)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)

    labels = pd.read_csv(LABELS_FILE_PATH)

    # Using a pretrained model to help accuracy of the neural network's predictions
    model = models.vgg16(pretrained=True)

    # Freeze pretrained model parameters to avoid backpropogating through them
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Creating a classifier for the specific dataset
    classifier = nn.Sequential(OrderedDict([
        ("fcl", nn.Linear(25088, 5000))
        ("relu", nn.ReLU()),
        ("drop", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(5000, 10222)),
        ("output", nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    # Gradient descent optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



if __name__ == "__main__":
    main()

def train_classifier():
    with active_session():
        epochs = 15
        steps = 0
        print_every = 40

        model.to("cuda")

        for e in range(epochs):
            model.train()
            running_loss = 0

def validation(model, testloader, criterion):
    val_loss = 0
    accuracy = 0

    for images, labels in iter(testloader):
        images, labels = images.to("cuda"), labels.to("cuda")
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        probabilities = torch.exp(output)
        equality = (labels.data == probabilities.max(dim-1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy