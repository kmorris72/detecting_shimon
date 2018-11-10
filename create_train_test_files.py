# Created by Kipp Morris Fall 2018 for use by Georgia Tech Robotic
# Musicianship VIP Group

# Script that splits the dataset into training and testing sets and then
# writes the names of the images to files called train.txt and test.txt
# that darknet looks at to know which images to use.


import glob
import os
import random


# Extensions of images files to look for
EXTENSION = ["jpg", "jpeg", "png"]

# The percentage of the images that should be designated for training
# For example, setting this to 0.8 would do an 80/20 training/testing split
TRAIN_PERCENTAGE = 0.8

img_filenames = []

# Get names of all image files
for ext in EXTENSION:
    img_filenames = img_filenames + glob.glob("darknet/all_images/*.{}".format(ext))

random.shuffle(img_filenames)

last_training_ind = int(TRAIN_PERCENTAGE * len(img_filenames))
train_file = open("darknet/train.txt", "w")
test_file = open("darknet/test.txt", "w")

for i in range(0, last_training_ind):
    train_file.write("{}\n".format(img_filenames[i]))
for i in range(last_training_ind, len(img_filenames)):
    test_file.write("{}\n".format(img_filenames[i]))
