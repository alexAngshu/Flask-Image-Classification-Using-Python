import numpy as np
import tensorflow as tf
import cv2, os

class Mobilenet:
    def __init__(self, folder_path):
        data = open(os.path.join(folder_path, 'labels_mobilenet.txt'),'r').read().split('\n')
        self.class_dict = {key: name for key, name in enumerate(data)}

        self.interpreter = tf.lite.Interpreter(model_path=os.path.join(folder_path, 'mobilenet_v1.tflite'))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        #print(input_details)
        #print(output_details)

        self.input_shape = self.input_details[0]['shape']

    def Predict(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        input_data = np.array(img, dtype=np.uint8).reshape(self.input_shape)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        index = np.argmax(output_data[0])
        accuracy = np.max(output_data[0])/np.sum(output_data[0])
        class_name = self.class_dict.get(index)
        return class_name, accuracy
