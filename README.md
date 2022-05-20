# small-target-tracker
Computer Vision Sem 1 2022 Project

## Introduction
This program tracks small objects from a sequence of satellite images found in 
https://github.com/The-Learning-And-VisionAtelier-LAVA/VISO 

## Procedure to run

1. Download the file from the github:
> https://github.com/jasperpato/small-target-tracker

2. Download the VISO file from the github:
> https://github.com/The-Learning-And-VisionAtelier-LAVA/VISO 

3. Download the requirements to run the program with:
> pip install -r requirements.txt

4. Run the program by running this line of code:
> python3 gui.py --dataset_path [file_path] --min_frame [start] --max_frame [finish]
>> #### Parameter info
>> - **file_path** must be a VISO/mot/[vehicle]/[file_number]
>> - **start** is the first frame to analyse. Must not be less than 1
>> - **finish** is the last frame to analyse. Must not be larger than the number of images in file_path

5. (Optional) Once the program launches, you can change the parameters in the text box.

6. Press start and wait for the tracker to load (Can take a while).

7. After loading, the program should show 4 graphs and a slideshow of the tracked objects.

