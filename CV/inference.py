import cv2
import numpy as np
from matplotlib import colors as clr
from matplotlib import pyplot as plt

num_none = 0
num_red = 0
num_yellow = 0
num_green = 0


def detect_traffic_light_state(img_bgr):
    global num_red, num_yellow, num_green, num_none

    # Convert Image to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, nchannel = img_rgb.shape
    channels = cv2.split(img_rgb)

    # Plot RGB Image in 2D Space
    plt.figure()
    plt.imshow(img_rgb)
    plt.title('RGB Image')
    plt.show()

    # Resize Image
    cut_off_percentage_height = 0.1
    cut_off_percentage_width = 0.15
    cut_of_height = int(cut_off_percentage_height * rows)
    cut_off_width = int(cut_off_percentage_width * cols)
    img_rgb = img_rgb[cut_of_height:rows-cut_of_height, cut_off_width: cols-cut_off_width]
    rows, cols, nchannel = img_rgb.shape

    plt.figure()
    plt.imshow(img_rgb)
    plt.title('RGB Image with Cut Offs')
    plt.show()

    # Plot RGB Image in 3D Space
    r, g, b = cv2.split(img_rgb)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection='3d')

    pixel_colors = img_rgb.reshape((np.shape(img_rgb)[0] * np.shape(img_rgb)[1], 3))
    norm = clr.Normalize(vmin=-1., vmax=1.)
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
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

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

    # Plot Histogram for HSV Image
    histogram = cv2.calcHist([img_hsv[:, :, 0]], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(histogram)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('HSV Image Histogram')
    plt.show()

    # Define ROIs
    red_lower1 = np.array([0, 80, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 80, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 30, 200])
    yellow_upper = np.array([40, 255, 255])

    green_lower1 = np.array([80, 50, 200])
    green_upper1 = np.array([100, 255, 255])
    green_lower2 = np.array([50, 50, 150])
    green_upper2 = np.array([80, 255, 255])

    # Create masks for Each Color
    red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
    green_mask1 = cv2.inRange(img_hsv, green_lower1, green_upper1)
    green_mask2 = cv2.inRange(img_hsv, green_lower2, green_upper2)
    green_mask = cv2.bitwise_or(green_mask1, green_mask2)

    # Apply Masks to the Original Images
    red_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)
    yellow_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
    green_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(red_roi)
    plt.subplot(1, 3, 2)
    plt.imshow(yellow_roi)
    plt.subplot(1, 3, 3)
    plt.imshow(green_roi)
    plt.show()

    # Close the Image
    closing_red = cv2.morphologyEx(red_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    closing_yellow = cv2.morphologyEx(yellow_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    closing_green = cv2.morphologyEx(green_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(closing_red)
    plt.subplot(1, 3, 2)
    plt.imshow(closing_yellow)
    plt.subplot(1, 3, 3)
    plt.imshow(closing_green)
    plt.show()

    # Count Weighted Pixel Sum
    sum_red = 0
    sum_green = 0
    sum_yellow = 0
    for i in range(rows):
        for j in range(cols):
            if np.any(closing_red[i, j, :] != [0, 0, 0]):
                if i < rows * 1/3:
                    sum_red += 3
                else:
                    sum_red += 1
            if np.any(closing_yellow[i, j, :] != [0, 0, 0]):
                if rows * 1/3 < i < rows * 2/3:
                    sum_yellow += 3
                else:
                    sum_yellow += 1
            if np.any(closing_green[i, j, :] != [0, 0, 0]):
                if i > rows * 2/3:
                    sum_green += 3
                else:
                    sum_green += 1

    print('Number of red pixels is: ', sum_red)
    print('Number of yellow pixels is: ', sum_yellow)
    print('Number of green pixels is: ', sum_green)

    sums = [sum_red, sum_yellow, sum_green]
    if max(sums) == 0:
        num_none += 1
        print('Class: light off')
    elif max(sums) == sum_red:
        num_red += 1
        print('Class: red light')
    elif max(sums) == sum_yellow:
        num_yellow += 1
        print('Class: yellow light')
    else:
        num_green += 1
        print('Class: green light')


if __name__ == '__main__':

    path = directory_path2 = 'C:/traffic_light_detection/CV/classification/none/none18.jpg'
    path2 = '/home/rtrk/teodora/traffic_light_detection/CV/classification2/red/red1.jpg'
    img = cv2.imread(path2)
    assert img is not None, "file could not be read, check with os.path.exists()"
    detect_traffic_light_state(img)
    # detect_lines(img)
