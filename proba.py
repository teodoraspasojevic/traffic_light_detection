import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5m")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
print('results beg')
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
results.save()
print('results end')

# Model parameters
print('Model parameters:')
for k, v in model.named_parameters():
    print(k)
