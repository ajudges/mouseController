'''
Head pose class
Authored by Nnamdi
'''
from model import Model_X
import cv2

class HeadPoseEstimation(Model_X):
    '''
    Head Pose Estimation Model child class with Model_X as parent class.
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.logger.info("preprocess input and start inference")
        
        p_image = self.preprocess_input(image)
        # sync inference
        self.logger.info("infer result")
        outputs=self.net.infer({self.input_name: p_image})
        self.logger.info("infered result")
        # wait for the result
        
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            outputs=self.net.requests[0].outputs
            
            return self.preprocess_output(outputs, image)

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        self.logger.info(" Getting the yaw, pitch, and roll angles ")
        angles = []
        
        angles.append(outputs['angle_y_fc'][0][0])
        angles.append(outputs['angle_p_fc'][0][0])
        angles.append(outputs['angle_r_fc'][0][0])
        
        cv2.putText(image, "Estimated yaw:{:.2f} | Estimated pitch:{:.2f}".format(angles[0],angles[1]), (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
        cv2.putText(image, "Estimated roll:{:.2f}".format(angles[2]), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0,255,0),1)
        return angles, image
