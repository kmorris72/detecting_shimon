# Created by Kipp Morris Fall 2018 for use by Georgia Tech Robotic
# Musicianship VIP Group

# Script that uses a JSON file from LabelBox to convert labels for object
# recognition from the format provided by LabelBox to the format used by
# darknet/YOLO.

# More specifically, for each image in the dataset, creates a text file
# with a line for each label in the image. Each line is of the following
# format:  <class number> <x> <y> <box width> <box height>
# ... where x is the x coordinate of the midpoint of the bounding box,
# y is the y coordinate of the midpoint of the bounding box, and box
# width and height are the corresponding dimensions of the bounding box.
# Note that each value except for class number is ***relative to the size
# of the image***.

# Also note that it is assumed that all images for the dataset being used
# can be found in a directory called "all_images" that is in the darknet
# directory.


import json
import os
import sys
from PIL import Image


# Mapping of class names to their class numbers (a requirement of darknet)
# Created using obj.names, which should be manually created and filled in
# with a class name on each line, where the line number is that class's number.
names_file = open("darknet/cfg/obj.names", "r")
class_nums = {}
i = 0
for obj_name in names_file.readlines():
    class_nums[obj_name.strip()] = i
    i += 1

try:
    json_filename = sys.argv[1]
except:
    raise Exception("Please provide the path of the json file containing the data labels as a command line argument.")

# JSON file is assumed to be in top level project directory with python scripts
json_file = open(json_filename, "r")
json_string = json_file.read()

# List of dictionaries
data = json.loads(json_string)

for item in data:
    img_path = item["External ID"]
    img_name = os.path.splitext(img_path)[0]

    # Open a text file corresponding to this image
    # Write bounding box (label) data to it in the format that darknet needs
    # try:
    text_file = open("darknet/all_images/{}.txt".format(img_name), "w")

    img = Image.open("darknet/all_images/{}".format(img_path))
    img_height = img.height
    img_width = img.width

    labels_dict = item["Label"]

    # Iterate over classes in image
    for class_name in labels_dict.keys():

        # Iterate over labels for a given class
        for label in labels_dict[class_name]:

            # List of dictionaries containing coordinates of bounding box
            # corners
            bb_coord = label["geometry"]


            # Labelbox doesn't seem to export the corners of the bounding box
            # in any specific order, so we have to semi-manually find the
            # coordinates of the specific corners...
            x_coords = []
            y_coords = []

            for corner in bb_coord:
                x_coords.append(corner["x"])
                y_coords.append(corner["y"])

            top_left_corner = (min(x_coords), min(y_coords))
            bottom_left_corner = (min(x_coords), max(y_coords))
            top_right_corner = (max(x_coords), min(y_coords))

            bb_width = top_right_corner[0] - top_left_corner[0]
            bb_height = bottom_left_corner[1] - top_left_corner[1]
            bb_rel_width = bb_width / img_width
            bb_rel_height = bb_height / img_height
            bb_x = (top_left_corner[0] + bb_width / 2) / img_width
            bb_y = (top_left_corner[1] + bb_height / 2) / img_height

            text_file.write("{} {} {} {} {}\n".format(class_nums[class_name], bb_x, bb_y, bb_rel_width, bb_rel_height))
    text_file.close()
