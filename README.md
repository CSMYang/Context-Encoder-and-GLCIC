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
    1. After you download raw dataset, firstly unzip it. Then, go GLCIC/make_dataset.py
    and run it directly since we have initialized all inputs.

    2. If you want to train again using other hyperparameters, go GLCIC/train.py and
    modify hyperparameters on args_dict.
       
    3. After you trained the model or used given traiend model, go GLCIC/predict.py to
    test images.

### Arguments
In GLCIC/subtitle_removal_predict.py:
* `<gpu>` (required): a boolean representing whether uses the GPU. Default is True.
* `<data_dir>` (required): path of the dataset directory.
* `<result_dir>` (required): path of the images to be stored during the training
* `<data_parallel>` (required): a boolean representing whether the data should be trained in parallel way. Default is True.
* `<recursive_search>` (required): a boolean representing whether the dataset should be se. Default is True.
* `<steps_1>` (required): path of the input directory including a set of frames/images
* `<steps_2>` (required): path of the output directory for generating a video
* `<steps_3>` (required): the width and height of the input frames/images, should be a tuple of two integers
* `<snaperiod_1>` (required): the number of frames per second for the output video, should be an int
* `<snaperiod_2>` (required): a boolean representing the method of generating mask for covering subtitles; True for the first method (a rectangle), False for the second method (subtitle itself)
* `<snaperiod_3>` (required): the image size for the model input, should be an int or a tuple of two integers
* `<max_holes>` (required): path of the input directory including a set of frames/images
* `<hole_min_w>` (required): path of the output directory for generating a video
* `<hole_max_w>` (required): the width and height of the input frames/images, should be a tuple of two integers
* `<hole_min_h>` (required): the number of frames per second for the output video, should be an int
* `<hole_max_h>` (required): a boolean representing the method of generating mask for covering subtitles; True for the first method (a rectangle), False for the second method (subtitle itself)
* `<cn_input_size>` (required): the image size for the model input, should be an int or a tuple of two integers
* `<ld_input_size>` (required): path of the input directory including a set of frames/images
* `<bsize>` (required): path of the output directory for generating a video
* `<bdivs>` (required): the width and height of the input frames/images, should be a tuple of two integers
* `<num_test_completions>` (required): the number of frames per second for the output video, should be an int
* `<mpv>` (required): a boolean representing the method of generating mask for covering subtitles; True for the first method (a rectangle), False for the second method (subtitle itself)
* `<alhpa>` (required): the image size for the model input, should be an int or a tuple of two integers
* `<arc>` (required): the image size for the model input, should be an int or a tuple of two integers


### Video Subtitle Removal:

    Note:
    We have provided a video on video directory. The make_video result as well as
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
    
### Arguments
In GLCIC/subtitle_removal_predict.py:
* `<model>` (required): path of the pretrained model (given)
* `<config>` (required): path of the model config file (given)
* `<input_img>` (required when testing one image): path of the image that need for subtitle removal
* `<output_img>` (required when testing one image): path of the output image
* `<input_img2>` (required when computing SSIM): path of the image that need to do comparison
* `<input_dir>` (required when testing a set of frames): path of the input directory including a set of frames/images
* `<output_dir>` (required when testing a set of frames): path of the output directory for generating a video
* `<video_size>` (required when testing a set of frames): the width and height of the input frames/images, should be a tuple of two integers
* `<fps>` (required when testing a set of frames): the number of frames per second for the output video, should be an int
* `<method>` (required): a boolean representing the method of generating mask for covering subtitles; True for the first method (a rectangle), False for the second method (subtitle itself)
* `<img_size>` (required): the image size for the model input, should be an int or a tuple of two integers

## References:

For the Context Encoder model, we wrote it based on the [code by the original author](https://github.com/pathak22/context-encoder).

For the GLCIC model, most are from [Otenim](https://github.com/otenim/GLCIC-PyTorch).

GLCIC/detect_subtitle.py:

For the subtitle removal method 1, we get the idea from [nathancy's answer](https://stackoverflow.com/questions/37771263/detect-text-area-in-an-image-using-python-and-opencv).

For the subtitle removal method 2, we get the idea from the [blog in programmersought](https://www.programmersought.com/article/5117975415/).

fid.py:

For calculating fid, we get the idea from [pytorch-fid source code](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py) and [Jason Brownlee's article](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/).
