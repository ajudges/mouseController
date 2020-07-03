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
import time
import os

logging.basicConfig(filename='mouseController.log', level=logging.DEBUG)


CPU_EXTENSION="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
performance_directory_path="../"
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
    optional.add_argument("-p", help="models precision e.g. FP16, FP32, INT8", required=False, default='FP32')
    optional.add_argument("-pd", help="path to store performance stats", required=False, default=performance_directory_path)

    args = parser.parse_args()

    return args

def pipelines(args):
    # enable logging for the function
    logger = logging.getLogger(__name__)
    
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
    start_face_model_load_time = time.time()
    faceDetectionPipeline=FaceDetection(faceDetectionModel, device, customLayers)
    faceDetectionPipeline.load_model()
    face_model_load_time = time.time() - start_face_model_load_time

    start_landmark_model_load_time = time.time()
    landmarksDetectionPipeline=FacialLandmarksDetection(landmarksDetectionModel, device, customLayers)
    landmarksDetectionPipeline.load_model()
    landmark_model_load_time = time.time() - start_landmark_model_load_time
    
    start_headpose_model_load_time = time.time()
    headPoseEstimationPipeline=HeadPoseEstimation(headPoseEstimationModel, device, customLayers)
    headPoseEstimationPipeline.load_model()
    headpose_model_load_time = time.time() - start_headpose_model_load_time
    
    start_gaze_model_load_time = time.time()
    gazeDetectionPipeline=Gaze(gazeDetectionModel, device, customLayers)
    gazeDetectionPipeline.load_model()
    gaze_model_load_time = time.time() - start_gaze_model_load_time
    
    
    # count the number of frames
    frameCount = 0

    # collate frames from the feeder and feed into the detection pipelines
    for _, frame in feed.next_batch():

        if not _:
            break
        frameCount+=1
        if frameCount%5==0:
            cv2.imshow('video', cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        start_face_inference_time = time.time()
        croppedFace = faceDetectionPipeline.predict(frame)
        face_inference_time = time.time() - start_face_inference_time

        if type(croppedFace)==int:
            logger.info("no face detected")
            if key==27:
                break
            continue
        
        start_landmark_inference_time = time.time()
        left_eye_image,right_eye_image, landmarkedFace = landmarksDetectionPipeline.predict(croppedFace.copy())
        landmark_inference_time = time.time() - start_landmark_inference_time

        if left_eye_image.any() == None or right_eye_image.any() == None:
            logger.info("image probably too dark or eyes covered, hence could not detect landmarks")
            continue
        
        cv2.imshow('Face output', landmarkedFace)

        start_headpose_inference_time = time.time()
        head_pose_angles=headPoseEstimationPipeline.predict(croppedFace.copy())
        
        headpose_inference_time = time.time() - start_headpose_inference_time
        
        start_gaze_inference_time = time.time()
        x,y=gazeDetectionPipeline.predict(left_eye_image ,right_eye_image, head_pose_angles)
        gaze_inference_time = time.time() - start_gaze_inference_time

        mouseVector=MouseController('medium','fast')


        if frameCount%5==0:
            mouseVector.move(x,y)

        if key==27:
            break
        
        output_path = performance_directory_path+args.p
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if face_inference_time != 0 and landmark_inference_time != 0 and headpose_inference_time != 0 and gaze_inference_time != 0:

            fps_face = 1/face_inference_time
            fps_landmark = 1/landmark_inference_time
            fps_headpose = 1/headpose_inference_time
            fps_gaze = 1/gaze_inference_time

            with open(os.path.join(output_path, 'face_stats.txt'), 'w') as f:
                f.write(str(face_inference_time)+'\n')
                f.write(str(fps_face)+'\n')
                f.write(str(face_model_load_time)+'\n')
            
            with open(os.path.join(output_path, 'landmark_stats.txt'), 'w') as f:
                f.write(str(landmark_inference_time)+'\n')
                f.write(str(fps_landmark)+'\n')
                f.write(str(landmark_model_load_time)+'\n')

            with open(os.path.join(output_path, 'headpose_stats.txt'), 'w') as f:
                f.write(str(headpose_inference_time)+'\n')
                f.write(str(fps_headpose)+'\n')
                f.write(str(headpose_model_load_time)+'\n')

            with open(os.path.join(output_path, 'gaze_stats.txt'), 'w') as f:
                f.write(str(gaze_inference_time)+'\n')
                f.write(str(fps_gaze)+'\n')
                f.write(str(gaze_model_load_time)+'\n')


        

        
        
    logger.info("The End")
    cv2.destroyAllWindows()
    feed.close()

def main():
    args=get_args()
    pipelines(args) 

if __name__ == '__main__':
    main()