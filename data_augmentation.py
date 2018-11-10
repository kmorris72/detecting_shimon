# Created by Kipp Morris Fall 2018 for use by Georgia Tech Robotic
# Musicianship VIP Group

# YOLO doesn't really require any image preprocessing, but this script will
# perform some different functions on the images to increase the size and
# difficulty of the dataset through data augmentation. Looks for images to use
# in a directory called "original_images" which should be in the darknet
# directory. Saves the resulting images in "all_images". Also saves a copy of
# each of the original images in "all_images".

# Could probably be improved with fancier functions and maybe a wider variety.
# Feel free to experiment.


from PIL import Image
from PIL import ImageFilter
import os


# Iteration Count
i = 1

img_names = os.listdir("darknet/original_images")
img_count = len(img_names)

print("Starting...", end=" ")

for filename in img_names:
    # Open original image
    img = Image.open("darknet/original_images/{}".format(filename))
    img.save("darknet/all_images/{}".format(filename))

    # Grayscale
    new_img = img.convert("L")
    new_img.save("darknet/all_images/grayscale_{}".format(filename))

    # Horizontal Flip
    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_img.save("darknet/all_images/hor_flip_{}".format(filename))

    # Vertical Flip
    new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_img.save("darknet/all_images/vert_flip_{}".format(filename))

    # Flip on Both Axes
    new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
    new_img.save("darknet/all_images/flip_both_{}".format(filename))

    # Brighten
    new_img = img.point(lambda p: p * 1.2)
    new_img.save("darknet/all_images/bright_{}".format(filename))

    # Darken
    new_img = img.point(lambda p: p * 0.8)
    new_img.save("darknet/all_images/dark_{}".format(filename))

    # Gaussian blur
    new_img = img.filter(ImageFilter.GaussianBlur())
    new_img.save("darknet/all_images/gauss_blur_{}".format(filename))

    if i % 10 == 0:
        print(">\nFinished with Image {}/{}".format(i, img_count), end=" ", flush=True)
    elif i == img_count:
        print(">\nDone!")
    else:
        print("-", end="", flush=True)

    i += 1
