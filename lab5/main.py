import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Define network architecture 1
# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(784, 512)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 256)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.relu1(self.fc1(x))
#         x = self.relu2(self.fc2(x))
#         x = self.fc3(x)
#         return x

# Define network architecture 2
# class Net(nn.Module):
#     def __init__(self, num_classes):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.maxpool(torch.relu(self.conv1(x)))
#         x = self.maxpool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Define network architecture 3
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.batch_norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batch_norm2(self.conv2(x))))
        x = self.pool3(self.relu3(self.batch_norm3(self.conv3(x))))
        x = x.view(-1, 64 * 3 * 3)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x



# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10
num_classes = 10

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion-MNIST dataset
train_dataset = FashionMNIST(root='./data/FashionMNIST', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data/FashionMNIST', train=False, transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = Net(num_classes=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
loss_history = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

        loss_history.append(loss.item())
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    confusion_matrix = np.zeros((num_classes,num_classes))
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i, label in enumerate(labels):
            confusion_matrix[label.item()][predicted[i].item()] += 1

    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: {} %'.format(accuracy))
    print('Confusion matrix:\n', confusion_matrix)
    
# Plot loss history
plt.plot(loss_history)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Find one misclassified image
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i, label in enumerate(labels):
            if predicted[i] != label:
                image = images[i]
                plt.imshow(image.squeeze(), cmap='gray')
                plt.title('Misclassified as {}'.format(predicted[i].item()))
                plt.show()
                break
        break

