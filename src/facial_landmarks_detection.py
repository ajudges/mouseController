'''
Facial landmark detection class
Authored by Nnamdi
'''

from model import Model_X
import cv2

class FacialLandmarksDetection(Model_X):
    '''
    Facial Landmarks Detection class with Model_X as parent class.
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)
    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        self.logger.info("preprocess input and perform inference ")
        
        p_image = self.preprocess_input(image)
        # synchronous inference
        self.net.infer({self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            outputs=self.net.requests[0].outputs[self.output_name]

            # select coords based on confidence threshold
            self.logger.info("Obtain coords of the left and right eyes")
            coords = self.preprocess_output(outputs)
            self.logger.info("Return coords and the cropped Face")
            return self.denorm_output(coords,image)

    # function to denormalize the output of the inference
    def denorm_output(self, coords,image):
        height = image.shape[0]
        width = image.shape[1]
        
        l_x0 = int(coords[0]*width) - 10
        l_x1 = int(coords[0]*width) + 10
        l_y0 = int(coords[1]* height) - 10
        l_y1 = int(coords[1]*height) + 10
        r_x0 = int(coords[2]*width) - 10
        r_x1 = int(coords[2]*width) + 10
        r_y0 = int(coords[3]*height) - 10
        r_y1 = int(coords[3]*height) + 10
        
        
        l_eye = image[l_y0:l_y1,l_x0:l_x1]
        r_eye = image[r_y0:r_y1,r_x0:r_x1]

        cv2.rectangle(image, (l_x0, l_y0), (l_x1, l_y1), (0, 0, 255), 2)
        cv2.rectangle(image, (r_x0, r_y0), (r_x1, r_y1), (0, 0, 255), 2)

        #cv2.imwrite("FacialLandmark.jpg", image)

        return l_eye,r_eye, image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # filter output based on confidence threshold
        outputs = outputs[0]
        self.logger.info("left and right eyes coordinates: {0}".format(outputs))
        xl=outputs[0][0]
        yl=outputs[1][0]
        xr=outputs[2][0]
        yr=outputs[3][0]
        return (xl,yl,xr,yr)
