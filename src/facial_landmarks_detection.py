'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class FacialLandmarksDetection:
    '''
    Class for Facial Landmarks Detection.
    '''
    def __init__(self, model_name, device='CPU', extensions=CPU_EXTENSION):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.cpu_extension=extensions

        try:
            self.model=IENetwork(self.model_structure,self.model_weights)
        except Exception as e:
            raise ValueError("Could not initialize the network. Have you entered the correct model path?")
        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core=IECore()

        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            ### TODO: Add any necessary extensions ###
            if self.cpu_extension and "CPU" in self.device:
                self.core.add_extension(self.cpu_extension, self.device)
            else:
                print("Add CPU extension and device type or run layer with original framework")
                exit(1)

        self.net=self.core.load_network(network=self.model,device_name=self.device,num_requests=1)

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        return

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        print('preprocess input')
        
        p_image = self.preprocess_input(image)
        # start asynchronous inference for specified request
        print('Start async inference')
        self.net.infer({self.input_name: p_image})
        
        # wait for the result
        if self.net.requests[0].wait(-1) == 0:
            # get the output of the inference
            print('Waiting for output of inference')
            outputs=self.net.requests[0].outputs[self.output_name]

            # select coords based on confidence threshold
            print('Obtain coords of the conf threshold')
            coords = self.preprocess_output(outputs)
            
            print('left_eye and right_eye')

            return self.denorm_output(coords,image)


    def denorm_output(self, coords,image):
        print("following coords are", coords)
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
        return l_eye,r_eye

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        dsize = (self.input_shape[3], self.input_shape[2])
        image = cv2.resize(image,(dsize))
        image = image.transpose((2,0,1))
        image = image.reshape(1,*image.shape)
        return image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # filter output based on confidence threshold
        outputs = outputs[0]
        print("here are the coords for landmark", outputs)
        xl=outputs[0][0]
        yl=outputs[1][0]
        xr=outputs[2][0]
        yr=outputs[3][0]
        return (xl,yl,xr,yr)
