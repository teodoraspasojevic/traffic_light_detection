import cv2
import numpy as np
from matplotlib import colors as clr
from matplotlib import pyplot as plt

num_red = 0
num_yellow = 0
num_green = 0


def detect_traffic_light_state(img_bgr):
    global num_red, num_yellow, num_green

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
    red_lower1 = np.array([0, 160, 200])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 160, 200])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([20, 100, 200])
    yellow_upper = np.array([30, 255, 255])

    green_lower1 = np.array([80, 50, 200])
    green_upper1 = np.array([100, 255, 255])
    green_lower2 = np.array([70, 100, 150])
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

    # Saturate Pixel Values
    for i in range(rows):
        for j in range(cols):
            if closing_red[i, j, 0] > 200:
                closing_red[i, j, :] = [255, 0, 0]
            else:
                closing_red[i, j, :] = [0, 0, 0]
            if closing_yellow[i, j, 0] > 200 and closing_yellow[i, j, 1] > 200:
                closing_yellow[i, j, :] = [255, 255, 0]
            else:
                closing_yellow[i, j, :] = [0, 0, 0]
            if closing_green[i, j, 1] > 200 or (closing_green[i, j, 1] > 100 and closing_green[i, j, 2] > 100):
                closing_green[i, j, :] = [0, 255, 0]
            else:
                closing_green[i, j, :] = [0, 0, 0]

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
    if max(sums) == sum_red:
        num_red += 1
        print('Class: red light')
    elif max(sums) == sum_yellow:
        num_yellow += 1
        print('Class: yellow light')
    else:
        num_green += 1
        print('Class: green light')


def detect_lines(img_bgr):
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, nchannel = img_rgb.shape

    # Detect Edges on the Image
    threshold1 = 200
    threshold2 = 400

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gray)
    plt.subplot(1, 2, 2)
    plt.imshow(edges)
    plt.show()

    # Detect horizontal lines
    threshold = 100
    minLineLength = 50
    maxLineGap = 10
    horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

    if horizontal_lines:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detected Lines', img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # Rotate the grayscale image by 90 degrees for vertical line detection
    # rotated_gray = np.rot90(gray)
    #
    # # Apply preprocessing (e.g., edge detection) on the rotated image
    # rotated_edges = cv2.Canny(rotated_gray, threshold1, threshold2)
    #
    # # Detect vertical lines
    # vertical_lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)
    #
    # # Rotate the detected vertical lines back to their original orientation
    # for line in vertical_lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(image, (y1, x1), (y2, x2), (0, 0, 255), 2)


if __name__ == '__main__':

    path = directory_path2 = 'C:/traffic_light_detection/CV/classification/green/green300.jpg'
    img = cv2.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    detect_traffic_light_state(img)
    detect_lines(img)
