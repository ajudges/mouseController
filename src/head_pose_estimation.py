'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=CPU_EXTENSION):
        '''
        TODO: Use this to set your instance variables.
        '''
        model_weight=model_name+'.bin'
        model_structure=model_name+'.xml'
        self.device=device
        self.cpu_extension=extensions
    
        try:
            self.model=IENetwork(model_structure,model_weight)
        except Exception as e:
            raise ValueError("Failed to load model. Kindly put the correct path")
        


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
        #self.output_name=[i for i in self.model.outputs.keys()]
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
            outputs=self.net.requests[0].outputs

            # select coords based on confidence threshold
            print('Obtain coords of the conf threshold')
            
            return self.preprocess_output(outputs)

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
        angles = []
        print(outputs)
        angles.append(outputs['angle_y_fc'][0][0])
        angles.append(outputs['angle_p_fc'][0][0])
        angles.append(outputs['angle_r_fc'][0][0])
        print('hey, here are the angles, dey sharp',angles)
        return angles
