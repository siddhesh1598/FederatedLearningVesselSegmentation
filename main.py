# -*- coding: utf-8 -*-
"""FL_Vessel_Segmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S2kINxYFRyrVzkLFFbgMCfxGQsiRl0qa
"""

# install syft if working on colab
!pip install syft

# unzip DRIVE and STARE datasets
!unzip /content/drive/My\ Drive/Colab\ Notebooks/VesselSegmentation/Source_DRIVE_datasets.zip
!unzip /content/drive/My\ Drive/Colab\ Notebooks/VesselSegmentation/Target_STARE_Dataset.zip

# import
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import syft as sy
import copy
import os

from model import UNet
from generator import DatasetGenerator
from util import *

# initilize directories of the datasets
DRIVE_imagesDir = "/content/Source_DRIVE_datasets/training/images"
DRIVE_masksDir = "/content/Source_DRIVE_datasets/training/1st_manual"

STARE_imagesDir = "/content/Target_STARE_Dataset/stare-images"
STARE_masksDir = "/content/Target_STARE_Dataset/labels-ah"

# initialze arguments and device
args = Arguments()
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize dataset dictionary and DataGen object
data_dirs = {
    "DRIVE": [DRIVE_imagesDir, DRIVE_masksDir],
    "STARE": [STARE_imagesDir, STARE_masksDir]
}

DataGen = DatasetGenerator(data_dirs, args)

# Generate dataset loader for DRIVE and STARE
DriveDatasetLoader = DataGen.generateTrainDataset("DRIVE")
StareDatasetLoader = DataGen.generateTrainDataset("STARE")

# plot image form the dataset
plot_images(DriveDatasetLoader, 0)

# define train function for DRIVE and STARE datasets
def train_model(trainDatasetLoader, model, optimizer, num_epochs=25):
    # keep track of best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(1, num_epochs + 1):
        # Set model to training mode
        model.train()  

        epoch_samples = 0

        for inputs, labels in trainDatasetLoader:
            # send data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = calculate_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            epoch_samples += inputs.size(0)

        epoch_loss = loss / epoch_samples

        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        # print loss
        if epoch % args.log_interval == 0:
            print("[INFO] Loss after {} epochs: {:.4f}".format(epoch,
                                                              epoch_loss.item()))

    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

# initialize models and optimizer
num_class = 3
model = UNet(num_class).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# train model on DRIVE dataset
print("Training model on DRIVE dataset...")
model_drive = train_model(DriveDatasetLoader, model, optimizer, num_epochs=args.epochs)

# train model on STARE dataset
print("Training model on STARE dataset...")
model_stare = train_model(StareDatasetLoader, model, optimizer, num_epochs=args.epochs)

# generate FL DataLoader and workers
FederatedDatasetLoader, workers = DataGen.generateFederatedDataset()

# get the workers
bob, alice, jon = workers # jon -> secure worker

# define train function for FL dataset
def train(args, model, device, train_loader, optimizer, epoch):

	# copy models and send it to workers
    model_bob = model.copy().send(bob)
    model_alice = model.copy().send(alice)

    # initialize optimizers for the models
    opt_bob = optim.Adam(model_bob.parameters(), lr=args.lr)
    opt_alice = optim.Adam(model_alice.parameters(), lr=args.lr)

    getModelOpt = {
    	model_bob.location: (model_bob, opt_bob), 
    	model_alice.location: (model_alice, opt_alice)
    }

    # set model to train
    model_bob.train()
    model_alice.train()

    epoch_samples = 0
    for batch_idx, (_data, _target) in enumerate(train_loader): 
        # send data to location
        (_model, _opt) = getModelOpt[_data.location]
        _data, _target = _data.to(device), _target.to(device)
        
        # forward
        _opt.zero_grad()
        output = _model(_data)
        loss = calculate_loss(output, _target)
        loss.backward()
        _opt.step()

        # statistics
        epoch_samples += _data.shape[0]
            
    # print loss
    if epoch % args.log_interval == 0:
        loss = loss.get()
        epoch_loss = loss / epoch_samples
        print("[INFO] Loss after {} epochs: {:.4f}".format(epoch,
                                                            epoch_loss.item()))
    
    return (model_alice, model_bob)

# initialize model for FL train
num_class = 3
model_fr = UNet(num_class).to(device)

# train the FL model
print("Training model on Federated dataset...")
for epoch in range(1, args.epochs+1):
    # get the models from the workers i.e. alice and bob
    modelA, modelB = train(args, model_fr, device, FederatedDatasetLoader, 
                           optimizer, epoch)
    # perform secure aggregation on the models
    model_fr = aggregate(model_fr, modelA, modelB, jon)

# generate dictionary of models
models = {
    "DRIVE": model_drive,
    "STARE": model_stare,
    "FEDERATED": model_fr
}

# generate test DataLoader
TestDatasetLoader = DataGen.generateTestDataset()

# define function to test the models on test dataset
def test(dataset, model, device, test_loader):
    # set the model to evaluation
    model.eval()
    iou_score = 0

    with torch.no_grad():

        for _data, _target in test_loader:
            # convert from 3-channel image to 1-channel image 
            _target = _target.sum(1, keepdim=True)
            # send data to evice
            _data, _target = _data.to(device), _target.to(device)
            
            # forward
            output = model.forward(_data)
            pred = output.argmax(1, keepdim=True)

            # convert to numpy
            _target = _target.cpu().numpy()
            pred = pred.cpu().numpy()

            # calculate IoU score
            iou_score += calculate_iou(pred, _target)

    # calculate average IoU score and print the score
    iou_score /= len(test_loader.dataset)
    
    print('{} set: \t{:.4f}'.format(dataset, iou_score))

# test the models
print("\t\tAverage IoU Score")
for (dataset, model) in models.items():
    test(dataset, model, device, TestDatasetLoader)

# test("DRIVE", model_drive, device, TestDatasetLoader)
# test("STARE", model_stare, device, TestDatasetLoader)
# test("FEDERATED LEARNING", model_fr, device, TestDatasetLoader)

# save models
print("Saving models...")
for (dataset, model) in models.items():
    PATH = os.path.sep.join(["models", "model_" + dataset + ".pt"])
    torch.save(model.state_dict(), PATH)