'''
Program to implement the various pipelines necessary for the pointer movement direction
'''

import cv2
import numpy as np
import argparse
import logging
from input_feeder import InputFeeder
from gaze_estimation import Gaze
from mouse_controller import MouseController
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from face_detection import FaceDetection


CPU_EXTENSION="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser()
    # -- Add required and optional groups
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- create the arguments
    optional.add_argument("-m_f", help="path to face detection model", default='../models/face-detection-adas-binary-0001', required=False)
    optional.add_argument("-m_l", help="path to facial landmarks detection model", default='../models/landmarks-regression-retail-0009', required=False)
    optional.add_argument("-m_h", help="path to head pose estimation detection model", default='../models/head-pose-estimation-adas-0001', required=False)
    optional.add_argument("-m_g", help="path to gaze detection model", default='../models/gaze-estimation-adas-0002',required=False)
    optional.add_argument("-l", help="MKLDNN (CPU)-targeted custom layers.", default=CPU_EXTENSION, required=False)
    optional.add_argument("-d", help="Specify the target device type", default='CPU')
    required.add_argument("-i", help="path to video/image file or 'cam' for webcam", required=True)

    args = parser.parse_args()

    return args

def pipelines(args):
    # enable logging for the function
    logger = logging.getLogger('pipelines')
    
    # grab the parsed parameters
    faceDetectionModel=args.m_f
    landmarksDetectionModel=args.m_l
    headPoseEstimationModel=args.m_h
    gazeDetectionModel=args.m_g
    device=args.d
    customLayers=args.l
    inputFile=args.i

    # initialize feed
    single_image_format = ['jpg','tif','png','jpeg', 'bmp']
    if inputFile.split(".")[-1].lower() in single_image_format:
        feed=InputFeeder('image',inputFile)
    elif args.i == 'cam':
        feed=InputFeeder('cam')
    else:
        feed = InputFeeder('video',inputFile)

    # load feed data
    feed.load_data()

    # initialize and load the models
    faceDetectionPipeline=FaceDetection(faceDetectionModel, device, customLayers)
    faceDetectionPipeline.load_model()

    landmarksDetectionPipeline=FacialLandmarksDetection(landmarksDetectionModel, device, customLayers)
    landmarksDetectionPipeline.load_model()
    
    headPoseEstimationPipeline=HeadPoseEstimation(headPoseEstimationModel, device, customLayers)
    headPoseEstimationPipeline.load_model()
    
    gazeDetectionPipeline=Gaze(gazeDetectionModel, device, customLayers)
    gazeDetectionPipeline.load_model()

    # break key
    key = cv2.waitKey(60)
    # count the number of frames
    frameCount = 0

    # collate frames from the feeder and feed into the detection pipelines
    for _, frame in feed.next_batch():
        if not _:
            break
        frameCount+=1
        if frameCount==10:
            print("Completed")
        croppedFace=faceDetectionPipeline.predict(frame.copy())
        if type(croppedFace)==int:
            logger.info("no face detected")
            if key==27:
                break
            continue
        left_eye_image,right_eye_image=landmarksDetectionPipeline.predict(croppedFace.copy())
        head_pose_angles=headPoseEstimationPipeline.predict(croppedFace.copy())
        if left_eye_image.any()!=None and right_eye_image.any()!=None:
            coord=gazeDetectionPipeline.predict(left_eye_image ,right_eye_image, head_pose_angles)
        else:
            exit(1)
        
        mouseVector=MouseController('high','fast')
        mouseVector.move(coord[0],coord[1])

        if key==27:
            break
        
    logger.info("The End")
    cv2.destroyAllWindows()
    feed.close()

def main():
    args=get_args()
    pipelines(args) 

if __name__ == '__main__':
    main()