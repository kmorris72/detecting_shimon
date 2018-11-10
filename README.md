# detecting_shimon: Created in Fall 2018 as a project for Georgia Tech's Robotic Musicianship VIP Group

## Explanation
The goal of this project was to create a pipeline of sorts through which VIP students can easily add new object classes to the YOLO object detection model. YOLO can be run using the neural network framework [Darknet](https://github.com/pjreddie/darknet), so this respository is meant to provide a streamlined method of getting data ready for use with Darknet.

## Dependencies
A short list of dependencies for my Python scripts:
* Numpy
```
pip install numpy
```
* OpenCV
```
pip install opencv-python
```
* Pillow
```
pip install Pillow
```

## Steps to Prepare Your Custom Data for Training

I found Darknet to be a little tricky to set up, so a good first step is to have YOLO train on just your data to make sure your setup is working. Here are the steps to take to get your data ready before doing that.

1. Clone this repository. Install Darknet into the project directory using instructions found [here](https://pjreddie.com/darknet/install/).

   Your root project directory should look something like this:

   IMAGE

2. Inside the darknet directory, create directories called "all_images" and "original_images". Put all of the images you wish to use for your object class in "original_images".

3. As an additional step here that may sound silly, check to make sure none of your image filenames contain duplicate or extra file extensions. It will cause so, *so* many problems later on, so just take the time now to be sure you're avoiding those problems.

4. If you wish to use data augmentation to increase the size of your dataset, data_augmentation.py provides a template for doing so. It will open each image in "original_images", save a copy in "all_images", and save copies augmented in different ways. Open it and check it out to see which augmentation methods you want to use. Add more or comment out the ones you don't want to use, and then run it from its location in the root project directory (no command line arguments needed) like so:
   ```
   python data_augmentation.py
   ```
   
5. In darknet/cfg, create these three files:
   * obj.names- Contains each of the class names from your data on separate lines, like this:
     ```
     Shimon
     Person
     Apple
     ...and so on
     ```
     
   * obj.data- Contains the following info that serves as a reference for Darknet about where and how your data is saved:
     ```
     classes= *The number of classes in your data here*  
     train  = train.txt  
     valid  = test.txt  
     names = obj.names  
     backup = backup/
     ```
   * ~.cfg- A CFG file corresponding to the YOLO model you plan to use. I used YOLO 2, so I used yolov2.cfg. Whichever one you're going to use, download it [here](https://pjreddie.com/darknet/yolo/).

6. If you don't already have one, make an account on [LabelBox](https://www.labelbox.com). There are a variety of tools out there for labeling images for object detection, but this is the one I chose to use. Create a dataset containing your images from "all_images", create a project, attach your dataset to it, and get labeling!

7. Once all of the images are labeled, export the labels in JSON format using the settings shown below.

   IMAGE
   
   Save the resulting JSON file in the root project directory.
   
8. From the root project directory, process the JSON file using process_json_labels.py like so:
   ```
   python process_json_labels.py json_filename
   ```
   Darknet requires the bounding box labels to be saved in a text file per image in the same directory as the images (more details in comments at the top of the script), so this script just reads in the JSON file and creates and saves those text files.
   
9. To verify that nothing has gone wrong with the labels, you can use label_test.py. Run it from the root project directory, providing the filename (not the full path, just the name of the image file itself) of the image you want to test with as a command line argument;
   ```
   python label_test.py image_filename
   ```
   The image will be displayed to you with the bounding boxes drawn and class names shown based on the information provided from the text files earlier so you can verify that they are correct. If there are no problems with your labels, you're done preparing your data!
