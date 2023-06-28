import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.yolo import Model

TRAIN_PATH = '/home/rtrk/teodora/datasets/traffic_light_real_world/train'
TEST_PATH = '/home/rtrk/teodora/datasets/traffic_light_real_world/test'

# Load the pre-trained YOLOv5 model
model = Model('/home/rtrk/teodora/traffic_light_detection/models/yolov5m.yaml')

# Freeze all layers except the last one
for param in model.parameters():
    param.requires_grad = False
# print(model.model)
print(type(model))                   # model - model.yolo.DetectionModel class - backbone (model.yolo.BaseModel) + head (model.yolo.Detect)
print(type(model.model))             # model.model - torch.nn.modules.container.Sequential - contains all of the layers
print(type(model.model[-1]))         # model.model[-1] - last layer - model.yolo.Detect - has ModuleList with three Conv2d layers
print(model.model[-1])
print(type(model.model[-1].m))       # model.model[-1].m - torch.nn.modules.container.ModuleList
print(model.model[-1].m)
print(type(model.model[-1].m[-1]))   # model.model[-1].m[-1] - last Conv2d in last layer
print(model.model[-1].m[-1].weight.requires_grad)

model.model[-1].m[-1].weight.requires_grad = True
model.model[-1].m[-1].bias.requires_grad = True

# Load the custom dataset
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transforms.ToTensor())
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
test_dataset = datasets.ImageFolder(TEST_PATH, transform=transforms.ToTensor())
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
