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
import time

## Sukhjit Kaur ##
## AMATH 582 Homework 5 ##

# Load and normalize the FashionMNIST dataset
train_batch_size = 512
test_batch_size = 256

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


# function to calculate weights
def calculate_fcn_weights(input_dim, hidden_layers, output_dim):
    total_weights = 0
    current_dim = input_dim
    for h in hidden_layers:
        total_weights += (current_dim * h) + h  # weights + biases
        current_dim = h
    total_weights += (current_dim * output_dim) + output_dim
    return total_weights


# Define FCN model class
class FCN(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, hidden_layers=[128, 64], dropout=0.0, batch_norm=True):
        super(FCN, self).__init__()
        layers = []
        current_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(current_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = h
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Define CNN model class
class CNN(nn.Module):
    def __init__(self, output_dim=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Convolutional layer 1
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization layer 1
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Convolutional layer 2
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization layer 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, output_dim)  # Fully connected layer 2
        self.dropout = nn.Dropout(0.5)  # Dropout layer

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))  # Apply conv1 -> bn1 -> ReLU -> pooling
       # print(f"After conv1: {x.shape}")
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))  # Apply conv2 -> bn2 -> ReLU -> pooling
        #print(f"After conv2: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        #print(f"After flattening: {x.shape}")
        x = nn.ReLU()(self.fc1(x))  # Fully connected layer with ReLU
        x = self.dropout(x)  # Dropout layer
        x = self.fc2(x)  # Output layer
        return x


# Function to count CNN weights
def count_cnn_weights(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Train and validate models
# def train_model(model, train_batches, val_batches, epochs, lr, optimizer_type, weight_init):
#     model_name = f'CNN_{sum(p.numel() for p in model.parameters())}_weights'
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr=lr)
#     train_loss_list, val_acc_list, train_acc_list, test_acc_list, results = [], [], [], [], []
#
#     for epoch in tqdm.trange(epochs):
#         model.train()
#         epoch_loss = 0
#         correct, total = 0, 0
#         for features, labels in train_batches:
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = loss_func(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         train_acc = correct / total
#         train_acc_list.append(train_acc)
#         train_loss_list.append(epoch_loss / len(train_batches))
#
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for features, labels in val_batches:
#                 features = features.view(-1, 28 * 28)  # Ensure this line for CNN (removing view for CNN usage)
#                 outputs = model(features)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         val_acc = correct / total
#         val_acc_list.append(val_acc)
#
#         # Test accuracy
#         correct, total = 0, 0
#         with torch.no_grad():
#             for features, labels in test_batches:
#                 features = features.view(-1, 28 * 28)  # Ensure this line for CNN
#                 outputs = model(features)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         test_acc = correct / total
#         test_acc_list.append(test_acc)
#
#         # Log results in .csv
#         results.append(
#             [model_name, optimizer_type, lr, weight_init, epoch_loss / len(train_batches), val_acc, test_acc])
#
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_batches):.4f}, Test Acc: {test_acc:.4f}")
#         if val_acc >= 0.880:
#             print(f'Success: Test Accuracy = {test_acc:.4f}')
#
#     # Save results to DataFrame
#     df_results = pd.DataFrame(results,
#                               columns=['Model', 'Optimizer', 'Learning Rate', 'Initialization', 'Training Loss',
#                                        'Validation Accuracy', 'Test Accuracy'])
#
#     df_results.to_csv(f'results_{model_name}_{optimizer_type}_{weight_init}_{lr}_{epochs}.csv', index=False)
#
#     print("Results saved.")
#     return train_loss_list, val_acc_list, train_acc_list

def train_model(model, train_batches, val_batches, epochs, lr, optimizer_type, weight_init):
    if isinstance(model, CNN):
        model_name = f'CNN_{sum(p.numel() for p in model.parameters())}_weights'
    if isinstance(model, FCN):
        model_name = f'FCN_{sum(p.numel() for p in model.parameters())}_weights'
    loss_func = nn.CrossEntropyLoss()
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

    train_loss_list, val_acc_list, train_acc_list, test_acc_list, results = [], [], [], [], []
    total_start_time = time.time()

    for epoch in tqdm.trange(epochs):
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0
        correct, total = 0, 0
        for features, labels in train_batches:
            if isinstance(model, CNN):
                # If model is an instance of CNN, ensure features are 2D image shape
                outputs = model(features)  # No flatten for CNN
            elif isinstance(model, FCN):
                # If model is an instance of FCN, flatten the features
                features = features.view(-1, 28 * 28)  # Flatten the input for FCN
            optimizer.zero_grad()  # reset gradient from previous batch
            outputs = model(features)  # Forward pass for FCN
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = correct / total
        train_acc_list.append(train_acc)
        train_loss_list.append(epoch_loss / len(train_batches))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in val_batches:
                if isinstance(model, CNN):
                    # If model is an instance of CNN, ensure features are in the right shape (2D image shape)
                    outputs = model(features)  # No need to flatten here for CNN
                elif isinstance(model, FCN):
                    # If model is an instance of FCN, flatten the features
                    features = features.view(-1, 28 * 28)  # Flatten the input for FCN
                outputs = model(features)  # Forward pass for FCN
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_acc_list.append(val_acc)

        # Test accuracy
        correct, total = 0, 0
        with torch.no_grad():
            for features, labels in test_batches:
                if isinstance(model, CNN):
                    # If model is an instance of CNN, ensure features are in the right shape (2D image shape)
                    outputs = model(features)  # No need to flatten here for CNN
                elif isinstance(model, FCN):
                    # If model is an instance of FCN, flatten the features
                    features = features.view(-1, 28 * 28)  # Flatten the input for FCN
                outputs = model(features)  # Forward pass for FCN
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        test_acc_list.append(test_acc)

        # Log results in .csv
        results.append(
            [model_name, optimizer_type, lr, weight_init, epoch_loss / len(train_batches), val_acc, test_acc])

        epoch_end_time = time.time()  # End time for the epoch
        epoch_time = epoch_end_time - epoch_start_time  # Time taken for the epoch
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_batches):.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f}s")
        # if test_acc >= 0.8800:
        #     print('Test accuracy of 88% reached.')
        #     total_end_time = time.time()  # End time for the total training
        #     total_time = total_end_time - total_start_time  # Total training time
        #     print(f"Total training time for {model_name}: {total_time:.2f} seconds")
        #     break
    total_end_time = time.time()  # End time for the total training
    total_time = total_end_time - total_start_time  # Total training time
    print(f"Total training time for {model_name}: {total_time:.2f} seconds")

    # Save results to DataFrame
    df_results = pd.DataFrame(results,
                              columns=['Model', 'Optimizer', 'Learning Rate', 'Initialization', 'Training Loss',
                                       'Validation Accuracy', 'Test Accuracy'])
    df_results['Training Time (s)'] = total_time
    df_results.to_csv(f'results_{model_name}_{optimizer_type}_{weight_init}_{lr}_{epochs}.csv', index=False)

    print("Results saved.")
    return train_loss_list, val_acc_list, train_acc_list, total_time

# Plot training loss and validation accuracy throughout the training epochs
def plot_metrics(model, train_loss_list, val_acc_list, epochs, lr, otype, intype):
    if isinstance(model, CNN):
        model_name = f'CNN_{sum(p.numel() for p in model.parameters())}_weights'
    if isinstance(model, FCN):
        model_name = f'FCN_{sum(p.numel() for p in model.parameters())}_weights'
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'{model_name} - {otype} + {intype}: {epochs} epochs with a learning rate of {lr}')
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

## Define CNN models with different weight constraints ##

class CNN_100k(CNN):
    def __init__(self, output_dim=10):
        super(CNN_100k, self).__init__(output_dim=output_dim)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 28, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(28)
        self.fc1 = nn.Linear(28 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, output_dim)

class CNN_50k(CNN):
    def __init__(self, output_dim=10):
        super(CNN_50k, self).__init__(output_dim=output_dim)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 14, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(14)
        self.fc1 = nn.Linear(14 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, output_dim)


# Model with ~20k weights
class CNN_20k(CNN):
    def __init__(self, output_dim=10):
        super(CNN_20k, self).__init__(output_dim=output_dim)
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 14, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(14)
        self.fc1 = nn.Linear(14 * 7 * 7, 24)
        self.fc2 = nn.Linear(24, output_dim)

# Model with ~10k weights
class CNN_10k(CNN):
    def __init__(self, output_dim=10):
        super(CNN_10k, self).__init__(output_dim=output_dim)
        self.conv1 = nn.Conv2d(1, 9, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 10, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(10 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, output_dim)

cnn_100k = CNN_100k()
cnn_50k = CNN_50k()
cnn_20k = CNN_20k()
cnn_10k = CNN_10k()

# # Check number of weights in CNN
# print("CNN with ~100k weights:", count_cnn_weights(cnn_100k))
# print("CNN with ~50k weights:", count_cnn_weights(cnn_50k))
# print("CNN with ~20k weights:", count_cnn_weights(cnn_20k))
# print("CNN with ~10k weights:", count_cnn_weights(cnn_10k))

# test to find weights
# print("FCN Weights:", calculate_fcn_weights(784, [200,150,75], 10))

# Main
epochs = 20
lr = 0.001  # 0.0005 or 0.01
optimizer_type = 'Adam'  # 'SGD' or 'RMSProp' or 'Adam'
init_type = 'kaiming'  # 'xavier' or 'random' or 'kaiming'
dropout_rate = 0.2 #
batch_norm = True  # Keep batch normalization enabled
#weight_decay = 0.001  # Add weight decay for L2 regularization

#FCN training

#Define FCN models with different weight constraints
fcn_100k = FCN(hidden_layers=[100,75,50], dropout=dropout_rate, batch_norm=batch_norm) # approx 100k weights
fcn_50k = FCN(hidden_layers=[50,40,30], dropout=dropout_rate, batch_norm=batch_norm) # approx 50k weights
fcn_200k = FCN(hidden_layers=[200,150,75], dropout=dropout_rate, batch_norm=batch_norm) #  approx 200k weights

train_loss, val_acc, train_acc, total_time = train_model(fcn_100k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(fcn_100k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)
#
train_loss, val_acc, train_acc, total_time = train_model(fcn_200k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(fcn_200k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)
#
train_loss, val_acc, train_acc, total_time = train_model(fcn_50k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(fcn_50k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)

#CNN training
train_loss, val_acc, train_acc, total_time = train_model(cnn_100k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(cnn_100k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)

train_loss, val_acc, train_acc, total_time = train_model(cnn_50k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(cnn_50k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)

train_loss, val_acc, train_acc, total_time = train_model(cnn_20k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(cnn_20k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)

train_loss, val_acc, train_acc, total_time = train_model(cnn_10k, train_batches, val_batches, epochs, lr, optimizer_type, init_type)
plot_metrics(cnn_10k, train_loss, val_acc, epochs, lr, optimizer_type, init_type)

