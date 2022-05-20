# small-target-tracker
Computer Vision Sem 1 2022 Project

## Introduction
This program tracks small objects from a sequence of satellite images found in 
https://github.com/The-Learning-And-VisionAtelier-LAVA/VISO 

## Procedure to run

1. Download the file from the github:
> https://github.com/jasperpato/small-target-tracker.git

2. Download the requirements to run the program with:
> pip install -r requirements.txt

3. Run the program by stating the necessary info in this format
>  gui.py [-h] --dataset_path DATASET_PATH [--show_blobs] [--min_frame MIN_FRAME]
>              [--max_frame MAX_FRAME]
   The dataset path must be a path leading to a VISO/mot/[vehicle]/[file number] 
   Example:
>  python3 gui.py -- dataset_path VISO/mot/car/001 --min_frame 1 --max_frame 100

4. (Optional) Once launch you can change the parameters in the text box

5. Press start and wait for the trracker to load (Can take a while)

6. After loading, the program should show 4 graphs and a slideshow of the tracked objects

