import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from collections import OrderedDict
from PIL import Image
import json

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 
    
    data_transforms = {"train": train_transforms,
                       "valid": valid_transforms,
                       "test":  test_transforms}
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data  = datasets.ImageFolder(test_dir,  transform=test_transforms )
    image_datasets = {"train": train_data,
                      "valid": valid_data,
                      "test":  test_data}

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=32)
    dataloaders = {"train": trainloader,
                   "valid": validloader,
                   "test":  testloader}
    
    return image_datasets, dataloaders

class feed_farwrd(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, drop_p=0.5):
        super().__init__()
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        # Forward through each hidden layer with ReLU activation and dropout
        for lay in self.hidden_layers:
            x = F.relu(lay(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# define a network with pre-trained model: densenet121
def Network(drop_p, hidden_layer_size, output_size, class_to_idx, arch='densenet121'):
    # load a pre-trained model: densenet121
    if (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif (arch == 'vgg16'):
        model = models.vgg16(pretrained=True) 
        input_size = model.classifier[0].in_features
    else:
        model = models.densenet121(pretrained=True)        
        input_size = model.classifier.in_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = feed_farwrd(input_size, hidden_layer_size, output_size, drop_p)
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    return model

# define function computing accuracy
def validation (model, testloader, criterion, device='cpu', print_flag=True):    
    ''' Validate the training pass 
    '''
    
    model.to(device)
    model.eval()
    
    test_loss = 0    
    accuracy  = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            #outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            #correct += (predicted == labels).sum().item()
            
            outputs = model.forward(images)
            test_loss += criterion(outputs, labels).item()
            
            ps = torch.exp(outputs).data
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
    
    if print_flag: 
        print('Test Loss: {:.3f}.. '.format(test_loss/len(testloader)),
              'Test Accuracy: {:.3f}'.format(accuracy/len(testloader)))
        
    return test_loss, accuracy

# define training process
def train (model, trainloader, validloader, criterion, optimizer, epochs, print_every, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda/cpu
    model.to(device)

    for e in range(epochs):
        model.train()
        
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                
                test_loss, accuracy = validation(model, validloader, criterion, device, False)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                
                model.train()
                
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])
                                        ]) 
    
    return img_transforms(img)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)

    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])
                                        ]) 
    
    return img_transforms(img)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
 
def load_checkpoint(filepath):
    ''' function that loads a checkpoint and rebuilds the model
    '''
    checkpoint = torch.load(filepath)
    
    model = Network(checkpoint['drop_p'], checkpoint['hidden_layer_size'],
                    checkpoint['output_size'], checkpoint['class_to_idx'], checkpoint['arch'])

    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, checkpoint, device='cuda', topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model = load_checkpoint(checkpoint)
    model.to(device)
    
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.float()
    with torch.no_grad():
        output = model.forward(img.to(device))
        
    output = F.softmax(output.data, dim=1)
    
    probs, indices = output.topk(topk)
    idx_to_class = {model.class_to_idx[k]:k for k in model.class_to_idx}

    classes = [idx_to_class[idx] for idx in indices.cpu().numpy()[0]]
    
    return np.array(probs)[0], classes

def check_sanity(image_path, probs, classes, cat_to_name):
    ''' 
        Viewing an image and it's predicted classes.
    '''
    img_filename = image_path.split('/')[-2]
    flower_name = cat_to_name[img_filename]
    img = Image.open(image_path)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)      
    ax1.set_title(flower_name)
    ax1.imshow(img)
    ax1.axis('off')
    
    y_pos = np.arange(len(probs))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([cat_to_name[x] for x in classes])
    ax2.invert_yaxis()
