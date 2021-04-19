# CSC413 Project: Analysis of GAN-based Image Inpainting Model: Context Encoder and GLCIC: Globally and Locally Consistent Image Completion

Note: For detailed explanation, please see our report.

Our implemention requires cuda.
Open the project under the main directory CSC413-Project, or it might ouput pathing error

## Authors: 
Jiahao Cheng 
Yi Wai Chow 
Zhiyuan Yang 

## Language version: 
Python 3.7+

## Required packages: 
numpy, matplotlib, scipy, sys, cv2 (opencv-python: 4.5.1.48+), torch, torchvision, skimage, tqdm, PTL, glob, keras

## Required dataset:
CelebA, please download the dataset from [this link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM).

## Required pretrained model for GLCIC:
[pretrained_model_cn](https://github.com/CSMYang/CSC413-Project/blob/main/GLCIC/pretrained_model_cn)
[config.json](https://github.com/CSMYang/CSC413-Project/blob/main/GLCIC/config.json)

## How to run:

### Context Encoder:
    1. xxx

    2. xxx
       
    3. xxx

    4. xxx

### GLCIC:
    1. xxx

    2. xxx
       
    3. xxx

    4. xxx

### Video Subtitle Removal:

    Note:
    We have given a video on video directory. The make_video result as well as
    outputs of some frames can be found in GLCIC/video and Testing Result/GLCIC.
    
    If you want to test a single frame/image, skip the first step.
    1. To remove subtitles from a video, we have provided a set of frames from 
    Prof. Jimmy Ba's Ted Talk (https://www.youtube.com/watch?v=j2HnL6T5J4w). 
    If you want to test other videos, please download by yourself. Then, go 
    subtitle_dataset.py, modify parameters properly and run build_dataset() 
    function to get a set of frames.

    For a set of frames:
    2. Go GLCIC/subtitle_removal_predict.py, modify "input_dir", "output_dir", 
    "video_size", "fps", and "method" in the args_dict so that it could run correctly. 
    Then, comment out the image prediction part and activate make_video() so it could br ran.
       
    3. The output video should be in the "output_dir" and is named as "test.avi". 
    It takes times depending on the number of images.

    For a single frame/image:
    2. Go GLCIC/subtitle_removal_predict.py, modify "input_img", "output_img", "method", 
    "img_size", and "input_ing2" if necessary. Then, run its file directly.

## Reference:

For the Context Encoder model, we get the idea from xxx. You can check it [here]().

For the GLCIC model, we get the idea from Otenim. You can check it [here](https://github.com/otenim/GLCIC-PyTorch).

For the subtitle removal method 1 in GLCIC/detect_subtitle.py, we get the idea from nathancy's answer. You can check it [here](https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv).

For the subtitle removal method 2 code in GLCIC/detect_subtitle.py, we get the idea from the blog in programmersought. You can check it [here]
(https://www.programmersought.com/article/5117975415/).
