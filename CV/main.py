import cv2
import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load Image

filename = '/home/rtrk/teodora/traffic_light_detection/runs_rw1/detect_test_ft_crops/crops/' \
           'traffic_light/000021_jpg.rf.673ce57687e8e188dfc2c8fe49812746.jpg'

img_bgr = cv2.imread(filename)
assert img_bgr is not None, "file could not be read, check with os.path.exists()"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rows, cols, nchannel = img_rgb.shape
channels = cv2.split(img_rgb)

# Plot RGB Image in 2D Space

plt.figure()
plt.imshow(img_rgb)
plt.title('RGB Image')
plt.show()

# Plot RGB Image in 3D Space

r, g, b = cv2.split(img_rgb)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')

pixel_colors = img_rgb.reshape((np.shape(img_rgb)[0] * np.shape(img_rgb)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker='.')
axis.set_xlabel('Red')
axis.set_ylabel('Green')
axis.set_zlabel('Blue')
plt.title('RGB Image')
plt.show()

# Calculate the Histograms for Each Channel

histograms = []
colors = ['r', 'g', 'b']
for channel, color in zip(channels, colors):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    histograms.append(histogram)
    plt.plot(histogram, color=color)

plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('RGB Image Histogram')
plt.show()

# Transform Image from BGR to HSV Color Space

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Plot HSV Image in 2D Space

plt.figure()
plt.imshow(img_hsv)
plt.title('HSV Image')
plt.show()

# Plot HSV Image in 3D Space

h, s, v = cv2.split(img_hsv)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker='.')
axis.set_xlabel('Hue')
axis.set_ylabel('Saturation')
axis.set_zlabel('Value')
plt.title('HSV Image')
plt.show()

# Plot histogram for HSV image

histogram = cv2.calcHist([img_hsv[:, :, 0]], [0], None, [256], [0, 256])

plt.figure()
plt.plot(histogram)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('HSV Image Histogram')
plt.show()

# Define ROIs

red_lower1 = np.array([0, 50, 50])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 50, 50])
red_upper2 = np.array([180, 255, 255])

green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])

yellow_lower = np.array([20, 50, 50])
yellow_upper = np.array([30, 255, 255])

# Create the masks for each color

red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)
green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)

# Apply the masks to the original image to extract the ROIs
red_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)
green_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)
yellow_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(red_roi)
plt.subplot(1, 3, 2)
plt.imshow(green_roi)
plt.subplot(1, 3, 3)
plt.imshow(yellow_roi)
plt.show()

# Close the Image

closing_red = cv2.morphologyEx(red_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
closing_green = cv2.morphologyEx(green_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
closing_yellow = cv2.morphologyEx(yellow_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(closing_red)
plt.subplot(1, 3, 2)
plt.imshow(closing_green)
plt.subplot(1, 3, 3)
plt.imshow(closing_yellow)
plt.show()
