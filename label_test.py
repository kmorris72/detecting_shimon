# Created by Kipp Morris Fall 2018 for use by Georgia Tech Robotic
# Musicianship VIP Group

# Script that uses the text file generated by process_json_label.py
# to verify that the labels have been formatted correctly. It does this
# by opening the text file and the image, drawing the bounding boxes on the
# image using the data from the text file, and displaying it to you so you
# can visually verify that the boxes look correct.

# Takes as a command line argument the name of the image you wish to test.

# Assumes that both the image file and the text file are in a directory
# named "all_images" in the darknet directory.


import numpy as np
import cv2
import os
import sys


# Constant for size of padding around class names in label boxes (in pixels)
TEXT_PADDING = 4

# Mapping of class numbers to their names. Created using obj.names, which
# should be manually created and filled in with a class name on each line, where the line number is that class's number.
names_file = open("darknet/cfg/obj.names", "r")
class_names = {}
i = 0
for obj_name in names_file.readlines():
    class_names[i] = obj_name.strip()
    i += 1

try:
    image_path = sys.argv[1]
except:
    raise Exception("Please provide the name of the image you wish to test as a command line argument (be sure to include the file extension).")

text_path = os.path.splitext(image_path)[0] + ".txt"

img = cv2.imread("darknet/all_images/{}".format(image_path), cv2.IMREAD_COLOR)
img_height, img_width = img.shape[0:2]

text_file = open("darknet/all_images/{}".format(text_path), "r")

# Iterate over the labels contained in the file, drawing the boxes one by one
for label in text_file.readlines():
    data = label.split()
    x_mid_rel, y_mid_rel, width_rel, height_rel  = float(data[1]), float(data[2]), float(data[3]), float(data[4])

    box_width = img_width * width_rel
    box_height = img_height * height_rel
    box_midpt = (x_mid_rel * img_width, y_mid_rel * img_height)

    top_left_corner = (int(box_midpt[0] - box_width / 2), int(box_midpt[1] - box_height / 2))

    bottom_right_corner = (int(top_left_corner[0] + box_width), int(top_left_corner[1] + box_height))

    # Draw the bounding box
    cv2.rectangle(img, top_left_corner, bottom_right_corner, (0, 0, 255), 2)

    # Get size of text for class name
    text_size = cv2.getTextSize(class_names[int(data[0])], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0)

    label_bot_right_corner = (top_left_corner[0] + text_size[0][0] + TEXT_PADDING * 2, top_left_corner[1] + text_size[0][1] + TEXT_PADDING * 2)

    # Draw a filled-in box to display the class name in
    cv2.rectangle(img, top_left_corner, label_bot_right_corner, (0, 0, 255), -1)

    name_write_location = (top_left_corner[0] + TEXT_PADDING, label_bot_right_corner[1] - TEXT_PADDING)

    # Write the name in the box just drawn
    cv2.putText(img, class_names[int(data[0])], name_write_location, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 0)

cv2.imshow("Label Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
