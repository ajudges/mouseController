'''
Gaze Estimation class
Authored by Nnamdi
'''
from model import Model_X

class Gaze(Model_X):
    '''
    Class for the Gaze Estimation Model with Model_X as parent class.
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)
        
    def predict(self, leftEye, rightEye, headPose):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.logger.info("preprocess input")
        left_eye_image = self.preprocess_input(leftEye)
        right_eye_image = self.preprocess_input(rightEye)
        # start asynchronous inference for specified request
        self.logger.info("start sync request")
        self.net.infer({'head_pose_angles':headPose, 'left_eye_image': left_eye_image, 'right_eye_image': right_eye_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            
            outputs=self.net.requests[0].outputs[self.output_name]
            
            return self.preprocess_output(outputs)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x = outputs[0][0]
        y = outputs[0][1]

        return x,y
