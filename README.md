# Computer Pointer Controller

This project - Computer Pointer Controller - moves the mouse pointer to the direction of the eye gaze. It does this by using a combination of 4 different computer vision models -face detection model, landmark detection model, head-pose estimation model, and gaze estimation. The final output, which is the x and y coordinates of the eye gaze, from the combined models is then fed to a mouse controller which moves the mouse pointer to the given coordinates.
![](https://youtu.be/YpIxPZf8WCQ)

## Project Set Up and Installation

1. Install OpenVINO (You can run this [script](https://github.com/Tob-iee/OpenVINO_installation) to automate the installation of OpenVINO)

2. Clone/download this repo.

3. Use the requirements.txt file to install the required packages, i.e.
```
pip3 install requirements.txt
```

4. Use the OpenVINO model downloader to download the following models:

  a. Face detection model
  ```
  python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
  ```
  
  b. Landmark regression model
  ```
  python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
  ```
  
  c. Head-pose estimation model
  ```
  python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
  ```
  
  d. Gaze estimation model
  ```
  python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
  ```

## Directory Structure 

![Alt text](https://github.com/ajudges/mouseController/blob/master/bin/directorystructure.png)

## Demo

From terminal, navigate to the src folder on the cloned directory, and run
```
python3 main.py -i ../bin/demo.mp4 \
-m_f <path to face detection model xml file> \
-m_l <path to landmark detection model xml file> \
-m_h <path to head-pose estimation model xml file> \
-m_g <path to gaze estimation model xml file>
```

## Documentation

The required command line arguments are:

1. -i, which can either be the path of the input video or ``cam`` for camera

2. -m_f, path to face detection model 

3. -m_l, path to landmark detection model

4. -m_h, path to head-pose estimation model

5. -m_g, path to gaze estimation model

The optional command line arguments are:

1. -l, path for MKLDNN (CPU)-targeted custom layers

2. -d, target device type e.g. CPU, FPGA

3. -p, path (in the cloned directory) to store performance statistics i.e. inference time, fps, and model loading time. 

4. -vf, specify flags from m_f, m_l, m_h, m_g e.g. -vf m_f m_l m_h m_g (seperate each flag by space) for visualization of the output of intermediate models

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
