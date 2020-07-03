# Computer Pointer Controller

Short introduction to the project
This project - Computer Pointer Controller - moves the mouse pointer to the direction of the eye gaze. It does this by using a combination of 4 different computer vision models -face detection model, landmark detection model, head-pose estimation model, and gaze estimation. The final output, which is the x and y coordinates of the eye gaze, from the combined models is then fed to a mouse controller which moves the mouse pointer to the given coordinates.

## Project Set Up and Installation

1. Install OpenVINO (You can run this [script](https://github.com/Tob-iee/OpenVINO_installation) to automate the installation of OpenVINO)

2. Clone/download this repo.

3. Use the requirements.txt file to install the required packages, i.e.
```
pip3 install requirements.txt
```

## Demo

From terminal, navigate to the src folder on the cloned directory, and run
```
python3 main.py -i ../bin/demo.mp4
```

## Documentation

The project contains FP32 intermediate representation (IR) files of the following models face detection model, landmark regression model, head-pose estimation model, and gaze estimation model. The models can be found in the models folder and their file paths are already specified in the code. 

The only required command line argument is -i, which can either be the path of the input video or CAM for camera. 

The optional arguments for the models include: -m_f, "path to face detection model"; -m_l, "path to landmark detection model"; -m_h, "path to head-pose estimation model"; -m_g, "path to gaze estimation model." 

Other optional arguments include: -l, "path for MKLDNN (CPU)-targeted custom layers; -d, "target device type e.g. CPU, FPGA"; -p, this is useful for specifying the precision of the models e.g. INT8, FP16, FP32, if changed from the default FP32; -pd, path to store performance statistics e.g. model loading time. 

On running the program, two visualizations pop-up to provide visuals on what the models are seeing. 

## Benchmarks
![Alt text](https://github.com/ajudges/mouseController/blob/master/bin/Inference%20Time.png)

![Alt text](https://github.com/ajudges/mouseController/blob/master/bin/FPS.png)

![Alt text](https://github.com/ajudges/mouseController/blob/master/bin/Model%20load%20time.png)

## Results

Of the four models, the face detection model has the most latency across the precision types. Hence, the combined inferencing speed of the four models is mostly dependent on that of the face detection model. 

It can also be seen that there is a general decrease in the processed frames per second with increase in precision. This can be attributed to the increase in floating point numbers with increase in precision, hence the calculations become more computational intensive.

FP32 precision gives better accuracy than the rest, the increased accuracy is more noticeable in the output for the gaze estimation. This could be as a result of the gaze estimation model being the last model before the final output to the mouse controller, hence, the losses of lower precisions are being built up from the first model down to the gaze estimation model.

### Edge Cases

Certain situations make the inferencing to fail. If the lighting conditions are poor, the application may not be able to detect the face, and should incase it detects the face and can't pick out the left and right eyes, a message is logged that the image is too dark or eyes are covered, hence it can't pick out the features.

Also, if there are multiple people in the frame, it takes the first detected face and uses it in the inferencing flow.
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
