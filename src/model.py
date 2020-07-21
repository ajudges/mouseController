'''
Base Class for mouse controller models
Authored by Nnamdi Ajah
'''
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import sys
import logging 

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Model_X:
    '''
    Parenr Class for the computer vision models.
    '''
    def __init__(self, model_name, device='CPU', extensions=CPU_EXTENSION):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.cpu_extension=extensions
        self.logger = logging.getLogger(__name__)

        try:
            self.model=IENetwork(self.model_structure,self.model_weights)
        except Exception as e:
            self.logger.exception("Could not initialize the Network. Have you entered the correct model path?")
        

    def load_model(self, name=None):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        # initialize the IECore interface
        self.core = IECore()

        ### TODO: Check for supported layers ###
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers)!=0:
            ### TODO: Add any necessary extensions ###
            if self.cpu_extension and "CPU" in self.device:
                self.core.add_extension(self.cpu_extension, self.device)
            else:
                self.logger.debug("Add CPU extension and device type or run layer with original framework")
                exit(1)

        # load the model
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        if name is not None:
            self.input_name=[i for i in self.model.inputs.keys()]
            self.input_shape=self.model.inputs[self.input_name[1]].shape
        else:
            self.input_name=next(iter(self.model.inputs))
            self.input_shape=self.model.inputs[self.input_name].shape

        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        return 

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