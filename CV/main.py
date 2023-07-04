import cv2
import numpy as np
from matplotlib import cm
from matplotlib import colors as clr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def detect_traffic_state(img_bgr):
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

    red_lower1 = np.array([0, 160, 200])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 160, 200])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])

    yellow_lower = np.array([20, 100, 200])
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

    # Clip pixel values

    for i in range(rows):
        for j in range(cols):
            if closing_red[i, j, 0] > 200:
                closing_red[i, j, :] = [255, 0, 0]
            else:
                closing_red[i, j, :] = [0, 0, 0]
            if closing_green[i, j, 1] > 200:
                closing_green[i, j, :] = [0, 255, 0]
            else:
                closing_green[i, j, :] = [0, 0, 0]
            if closing_yellow[i, j, 0] > 200 and closing_yellow[i, j, 1] > 200:
                closing_yellow[i, j, :] = [255, 255, 0]
            else:
                closing_yellow[i, j, :] = [0, 0, 0]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(closing_red)
    plt.subplot(1, 3, 2)
    plt.imshow(closing_green)
    plt.subplot(1, 3, 3)
    plt.imshow(closing_yellow)
    plt.show()

    # Count pixel number

    sum_red = 0
    sum_green = 0
    sum_yellow = 0
    for i in range(rows):
        for j in range(cols):
            if np.any(closing_red[i, j, :] != [0, 0, 0]):
                sum_red += 1
            if np.any(closing_green[i, j, :] != [0, 0, 0]):
                sum_green += 1
            if np.any(closing_yellow[i, j, :] != [0, 0, 0]):
                sum_yellow += 1

    print('Number of red pixels is: ', sum_red)
    print('Number of green pixels is: ', sum_green)
    print('Number of yellow pixels is: ', sum_yellow)


def detect_lines(img_bgr):

    threshold1 = 50
    threshold2 = 150

    rows, cols, nchannel = img_bgr.shape

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing (e.g., edge detection)
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Detect horizontal lines
    threshold = 100
    minLineLength = 0.7*cols
    maxLineGap = 10
    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

    # Rotate the grayscale image by 90 degrees for vertical line detection
    rotated_gray = np.rot90(gray)

    # Apply preprocessing (e.g., edge detection) on the rotated image
    rotated_edges = cv2.Canny(rotated_gray, threshold1, threshold2)

    # Detect vertical lines
    vertical_lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

    # Rotate the detected vertical lines back to their original orientation
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (y1, x1), (y2, x2), (0, 0, 255), 2)

    # Draw the detected horizontal lines on the image
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the image with the detected lines
    cv2.imshow('Detected Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load Image

    filename = 'C:/traffic_light_detection/runs_rw1/detect_test_ft_crops/crops/' \
               'traffic_light/000602_jpg.rf.7fe4c75cb99948054d34bc54fc9106e02.jpg'
    img = cv2.imread(filename)
    assert img is not None, "file could not be read, check with os.path.exists()"
    detect_traffic_state(img)
