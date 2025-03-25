import numpy as np
import torch
from torch import nn, optim
import tqdm
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

## Sukhjit Kaur ##
## AMATH 582 Homework 4 - FashionMNIST Fully Connected Neural Networks ##

# Load and normalize the FashionMNIST dataset
train_batch_size=512
test_batch_size=256
train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                      ]))

test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                     ]))

train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)), train_dataset.targets, stratify=train_dataset.targets, test_size=0.1)

train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)

train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Define FCN model
class ACAIGFCN(nn.Module):
    # initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim=784, output_dim=10, hidden_layers=[128, 64], dropout=0.5, batch_norm=False):
        super(ACAIGFCN, self).__init__()
        # define the network layer(s) and activation function(s)
        layers = []
        current_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h)) #Fully Connected layer, start with 784
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))  # If batch normalization is true, add
            layers.append(nn.ReLU()) #reLU activation function
            layers.append(nn.Dropout(dropout)) #regularization - dropout factor of 0.5
            current_dim = h
        layers.append(nn.Linear(current_dim, output_dim)) #Fully Connected layer to map from last hidden layer to output layer of 10, for the 10 class types
        self.model = nn.Sequential(*layers) #sequential container containing all the layers, that is used to pass the input data forward

    def forward(self, x):
        #define how your model propagates the input through the network
        return self.model(x) #the input data gets moved forward through the network


# Train and validate model
def train_model(model, train_batches, val_batches, epochs, lr, optimizer_type, weight_init):
    loss_func = nn.CrossEntropyLoss()  # use cross entropy for loss
    #determine optimizer to use based on input parameter
    if optimizer_type == 'SGD':

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    elif optimizer_type == 'Adam':

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif optimizer_type == 'RMSProp':

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    #determine weight to use depending on input
    if weight_init == 'random':

        for m in model.modules():

            if isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, mean=0, std=0.01)

    elif weight_init == 'xavier':

        for m in model.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_normal_(m.weight)

    elif weight_init == 'kaiming':

        for m in model.modules():

            if isinstance(m, nn.Linear):

                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    train_loss_list, val_acc_list, test_acc_list, train_acc_list, results = [], [], [], [], []

    for epoch in tqdm.trange(epochs): #loop over all the epoch values
        model.train() #set model to train
        epoch_loss = 0 #loss for each epoch value looped
        correct, total = 0, 0
        for features, labels in train_batches: #loop over all the batches of data
            features = features.view(-1, 28 * 28) #reshape 28x28 to 784
            optimizer.zero_grad() #reset gradient from previous batch
            outputs = model(features) #pass forward to get prediction from the model
            loss = loss_func(outputs, labels) #compute loss by comparing predicted labels to true labels
            loss.backward() #backprop to get gradient for each parameter
            optimizer.step() #update weights based on the gradient and learning rate
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # returns the class with the highest 'score'
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # countup all correct predictions
        train_acc = correct / total
        train_acc_list.append(train_acc)
        train_loss_list.append(epoch_loss / len(train_batches)) #loss for epoch (how well the model fits the training data)

        # Calculate accuracy on test set -
        # Validation accuracy (can detect overfitting or underfitting)
        model.eval() #set model to validate - no dropout layers
        correct, total = 0, 0
        with torch.no_grad(): #no gradient computation done during validation
            for features, labels in val_batches:
                features = features.view(-1, 28 * 28)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1) #returns the class with the highest 'score'
                total += labels.size(0)
                correct += (predicted == labels).sum().item() #countup all correct predictions
        val_acc = correct / total
        val_acc_list.append(val_acc) #percentage of correct predictions

        # Test accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in test_batches:
                features = features.view(-1, 28 * 28)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        test_acc_list.append(test_acc)

        # Log results in .csv
        results.append([optimizer_type, lr, weight_init, epoch_loss / len(train_batches), val_acc, test_acc])

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_batches):.4f}, Val Acc: {val_acc:.4f}")

        # Save results to DataFrame
    df_results = pd.DataFrame(results, columns=['Optimizer', 'Learning Rate', 'Initialization', 'Training Loss',
                                                'Validation Accuracy', 'Test Accuracy'])

    df_results.to_csv(f'results_{optimizer_type}_{weight_init}_{lr}_{epochs}.csv', index=False)

    print("Results saved.")

    return train_loss_list, val_acc_list, train_acc_list


# Plot training loss and validation accuracy throughout the training epochs
def plot_metrics(train_loss_list, val_acc_list, epochs, lr, otype, intype):
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'{otype} + {intype}: {epochs} epochs with a learning rate of {lr}')
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

# Sample code to visulaize the first sample in first 16 batches
# batch_num = 0
# for train_features, train_labels in train_batches:
#
#      if batch_num == 16:
#         break    # break here
#
#      batch_num = batch_num +1
#      print(f"Feature batch shape: {train_features.size()}")
#      print(f"Labels batch shape: {train_labels.size()}")
#
#      img = train_features[0].squeeze()
#      label = train_labels[0]
#      plt.imshow(img, cmap="gray")
#      plt.show()
#      print(f"Label: {label}")

# Sample code to plot N^2 images from the dataset
def plot_images(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))

    for i in range(N):
        for j in range(N):
            ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
            ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
plt.show()

# Main
#plot_images(train_dataset.data[:64], 8, "First 64 Training Images")
#Test different weight initializations, dropout, and batch normalization
model = ACAIGFCN(hidden_layers=[256, 128, 64], dropout=0.5, batch_norm=True)
init_type = 'random'  #'random', 'xavier', or 'kaiming'
optimizer_type = 'Adam'  #optimizers: 'SGD', 'Adam', 'RMSProp'
lr = 0.01  #learning rate
epochs = 20 #epochs
train_loss, val_acc, train_acc = train_model(model, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
# plot_metrics(train_loss, val_acc, epochs, lr, optimizer_type, init_type)

# plotting of training vs val acc
# plt.figure(figsize=(12, 5))
# plt.title(f'With Batch Normalization and Dropout = 0.5')
# plt.plot(train_acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()



