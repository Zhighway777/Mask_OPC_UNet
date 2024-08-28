import os, time, sys, argparse
import numpy as np
from utils import str2bool, dir_parser

import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler

# Initialize the parser
parser = argparse.ArgumentParser(description='PyTorch Training Script')
# Optional arguments
parser.add_argument('--gpu', type=str2bool, nargs='?', const=True, default=torch.cuda.is_available(),
                    help='Use GPU or not (default: %(default)s)')
parser.add_argument("--load_pretrained_model", type=str, default=None)
parser.add_argument("--beta",type=float, default=1.45)
# Parse the arguments
args = parser.parse_args()

class Neural_ILT:
    def __init__(self, args):
    # set up basic para
    self.device = torch.device('cuda' if args.gpu else 'cpu')

    #litho-sim kernels init

    #init the unet and parse in the pretrained weights
    self.unet = UNet(n_channels=1, n_classes=1).to(self.device)
    self.unet.load_state_dict("path", map_location=self.device)

    #init the Neural-ILT backbone 
    self.refine_model = ILTNet(

    ).to(self.device)

    #init the complexity refinement layer
    self.cplx_loss_layer = ilt_loss(

    ).to(self.device)

    #prase the pretrained unet into neural-ilt
    pretrain_dictionary = self.unet.state.dict()
    self.model_dictionary = self.refine_model.state.dict()
    pretrain_dictionary = {k: v for k, v in pretrain_dictionary.items() if k in self.model_dictionary}

    for item in self.refine_model.parameters():
        item.requires_grad = True

    self.model_dictionary.update(pretrain_dictionary)
    self.refine_model.load_state_dict(self.model_dictionary)
    

    #optimizer is adam
    self.optimizer_ft = optim.Adam(self.refine_model.parameters(), lr=2e-3)
# Load the dataset
train_loader, val_loader = dir_parser()

# Define the model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # Assuming 10 classes

# Set the model to evaluation mode
model.eval()

# Set the device for the model
model.to(device)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
for epoch in range(10):
    for data, target in train_loader:
        # Send data and target to the device
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the weights
        optimizer.step()

    # Update the learning rate
    scheduler.step()

    # Validation loop
    for data, target in val_loader:
        # Send data and target to the device
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)

        # Calculate the loss
        loss = criterion(output, target)

    # Print the loss
    print(f'Epoch: {epoch+1}, Training Loss: {loss.item():.4f}')


