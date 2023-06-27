import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.yolo import Model

# Load the pre-trained YOLOv5 model
model = Model('yolov5m.yaml')

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False
model.model[-1].conv[-1].weight.requires_grad = True
model.model[-1].conv[-1].bias.requires_grad = True

# Load the custom dataset
train_dataset = datasets.ImageFolder('path/to/train/dataset', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train the last layer of the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.model[-1].conv[-1].parameters(), lr=0.001)
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
test_dataset = datasets.ImageFolder('path/to/test/dataset', transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print('Accuracy: {:.2f}%'.format(accuracy))
