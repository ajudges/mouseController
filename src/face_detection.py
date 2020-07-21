'''
Face detection class
Authored by Nnamdi
'''

from model import Model_X

class FaceDetection(Model_X):
    '''
    Face Detection Model child class with Model_X as parent class.
    '''
    def __init__(self, model_name, device, extensions):
        super().__init__(model_name, device, extensions)
        

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        # preprocess the image
        self.logger.info("preprocess input and start inference")
        
        p_image = self.preprocess_input(image)
        # start synchronous inference for specified request
        self.net.infer({self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            self.logger.info("Waiting for output of inference")
            outputs=self.net.requests[0].outputs[self.output_name]

            # select coords based on confidence threshold
            coords = self.preprocess_output(outputs)
            
            self.logger.info("cropped face: {0}".format(coords))
            return self.crop_output(coords,image)

    def crop_output(self, coords, image):
        height = image.shape[0]
        width = image.shape[1]
        
        for x1, y1, x2, y2 in coords:
            
            #conf = box[2]
            #if conf > self.threshold:
            xmin = int(x1 * width)
            ymin = int(y1 * height)
            xmax = int(x2 * width)
            ymax = int(y2 * height)
            image = image[ymin:ymax,xmin:xmax]
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            #cv2.imwrite("ComputerPointerWindow.jpg", image)
            #out_coord.append([xmin,ymin,xmax,ymax])
        return image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # filter output based on confidence threshold
        coords = []
        for box in outputs[0][0]:
            conf = box[2]
            if conf > 0.6:
                coords.append(box[3:])
        return coords