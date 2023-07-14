import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

num_red = 0
num_yellow = 0
num_green = 0
num_none = 0

# Number of images in dataset1
# total_red = 618
# total_yellow = 16
# total_green = 421
# total_none = 39

# Number of images in dataset2
total_red = 432
total_yellow = 31
total_green = 149
total_none = 826

conf_matrix = np.zeros((4, 4))


def detect_traffic_light_state(img_bgr, label):
    global num_red, num_yellow, num_green, num_none, conf_matrix

    # Convert Image to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, nchannel = img_rgb.shape
    channels = cv2.split(img_rgb)

    # Cutoff Image
    cut_off_percentage_height = 0.1
    cut_off_percentage_width = 0.15
    cut_of_height = int(cut_off_percentage_height * rows)
    cut_off_width = int(cut_off_percentage_width * cols)
    img_rgb = img_rgb[cut_of_height:rows-cut_of_height, cut_off_width: cols-cut_off_width]
    rows, cols, nchannel = img_rgb.shape

    # Transform Image from RGB to HSV Color Space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define ROIs
    red_lower1 = np.array([0, 50, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 100])
    red_upper2 = np.array([180, 255, 255])

    yellow_lower = np.array([10, 30, 200])
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

    # Close the Image
    closing_red = cv2.morphologyEx(red_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    closing_yellow = cv2.morphologyEx(yellow_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    closing_green = cv2.morphologyEx(green_roi, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))

    # Count Weighted Pixel Sum
    sum_red = 0
    sum_yellow = 0
    sum_green = 0
    for i in range(rows):
        for j in range(cols):
            if np.any(closing_red[i, j, :] != [0, 0, 0]) and rows*0.2 < i < rows*0.8 and cols*0.2 < j < cols*0.8:
                if i < rows * 1/3:
                    sum_red += 3
                else:
                    sum_red += 1
            if np.any(closing_yellow[i, j, :] != [0, 0, 0]) and rows*0.2 < i < rows*0.8 and cols*0.2 < j < cols*0.8:
                if rows * 1/3 < i < rows * 2/3:
                    sum_yellow += 3
                else:
                    sum_yellow += 1
            if np.any(closing_green[i, j, :] != [0, 0, 0]) and rows*0.2 < i < rows*0.8 and cols*0.2 < j < cols*0.8:
                if i > rows * 2/3:
                    sum_green += 3
                else:
                    sum_green += 1

    # Classify and Save Image
    output_directory = 'C:/traffic_light_detection/CV/classification2'

    sums = [sum_red, sum_yellow, sum_green]
    if max(sums) == 0:
        num_none += 1
        output_directory = os.path.join(output_directory, 'none')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_filename = 'none' + str(num_none) + '.jpg'
        conf_matrix[3, label] += 1
    elif max(sums) == sum_red:
        num_red += 1
        output_directory = os.path.join(output_directory, 'red')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_filename = 'red' + str(num_red) + '.jpg'
        conf_matrix[0, label] += 1
    elif max(sums) == sum_yellow:
        num_yellow += 1
        output_directory = os.path.join(output_directory, 'yellow')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_filename = 'yellow' + str(num_yellow) + '.jpg'
        conf_matrix[1, label] += 1
    else:
        num_green += 1
        output_directory = os.path.join(output_directory, 'green')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_filename = 'green' + str(num_green) + '.jpg'
        conf_matrix[2, label] += 1
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, img_bgr)


if __name__ == '__main__':

    directories1 = ['C:/traffic_light_detection/CV/dataset1/red', 'C:/traffic_light_detection/CV/dataset1/yellow',
                   'C:/traffic_light_detection/CV/dataset1/green', 'C:/traffic_light_detection/CV/dataset1/none']
    directories2 = ['C:/traffic_light_detection/CV/dataset2/red', 'C:/traffic_light_detection/CV/dataset2/yellow',
                   'C:/traffic_light_detection/CV/dataset2/green', 'C:/traffic_light_detection/CV/dataset2/none']
    directories3 = ['/home/rtrk/teodora/traffic_light_detection/CV/dataset2/red', '/home/rtrk/teodora/traffic_light_detection/CV/dataset2/yellow',
                   '/home/rtrk/teodora/traffic_light_detection/CV/dataset2/green', '/home/rtrk/teodora/traffic_light_detection/CV/dataset2/none']
    classes = [0, 1, 2, 3]

    for directory_path, label in zip(directories2, classes):
        for filename in os.listdir(directory_path):
            if filename.endswith('.jpg'):
                file_path = os.path.join(directory_path, filename)

                img = cv2.imread(file_path)
                assert img is not None, "file could not be read, check with os.path.exists()"

                detect_traffic_light_state(img, label)

    conf_matrix[0, 0] /= total_red
    conf_matrix[1, 0] /= total_red
    conf_matrix[2, 0] /= total_red
    conf_matrix[3, 0] /= total_red
    conf_matrix[0, 1] /= total_yellow
    conf_matrix[1, 1] /= total_yellow
    conf_matrix[2, 1] /= total_yellow
    conf_matrix[3, 1] /= total_yellow
    conf_matrix[0, 2] /= total_green
    conf_matrix[1, 2] /= total_green
    conf_matrix[2, 2] /= total_green
    conf_matrix[3, 2] /= total_green
    conf_matrix[0, 3] /= total_none
    conf_matrix[1, 3] /= total_none
    conf_matrix[2, 3] /= total_none
    conf_matrix[3, 3] /= total_none

    conf_matrix = np.round(conf_matrix, 3)

    print('Final number of  detected traffic lights with red light: ', num_red)
    print('Final number of  detected traffic lights with yellow light: ', num_yellow)
    print('Final number of  detected traffic lights with green light: ', num_green)
    print('Confusion matrix: ', conf_matrix)

    # Plotting the confusion matrix as a heatmap
    plt.imshow(conf_matrix, cmap='Blues')

    plt.colorbar()
    plt.xlabel('True')
    plt.ylabel('Predicted')
    tick_marks = np.arange(len(conf_matrix))
    plt.xticks(tick_marks, ['red', 'yellow', 'green', 'none'])
    plt.yticks(tick_marks, ['red', 'yellow', 'green', 'none'])
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

    plt.savefig('confusion_matrix2.png', format='png')
    plt.show()
