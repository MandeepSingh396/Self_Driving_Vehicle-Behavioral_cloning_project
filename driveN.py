#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

#load our saved model
import torch
from torch.autograd import Variable
# from mymodel import *
import torchvision.transforms as transforms
#helper class
import utils
from torchvision.transforms.functional import crop

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
#init our model and image array as empty
model = None
prev_image_array = None

#set min/max speed for our autonomous car
MAX_SPEED = 15
MIN_SPEED = 5

#and a speed limit
speed_limit = MAX_SPEED
# transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])
def cropimg(image):
    return crop(image,70,0,65,320)
transformations = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(cropimg),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        
        # The current image from the center camera of the car
        original_image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            # image = np.asarray(original_image)       # from PIL image to numpy array
            # image = utils.preprocess(image) # apply the preprocessing
            image = transformations(original_image)
            # image = torch.Tensor(image)
            #image = np.array([image])       # the model expects 4D array

            image = image.view(1, 3, 65, 320)
            image = Variable(image)
            
            # predict the steering angle for the image
            steering_angle = model(image).view(-1).data.numpy()[0]
            
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            #throttle = controller.update(float(speed)) - 0.1
            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print("Exception")
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            original_image.save('{}.jpg'.format(image_filename))
    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

import torch.nn as nn
import torch.nn.functional as F

class Nvidia(nn.Module):
    def __init__(self):
        super(Nvidia, self).__init__()
        
        self.conv1 = nn.Conv2d(3,24,5, padding = 0, stride=2) #in-channels, out-channels, kernel_size
        self.conv2 = nn.Conv2d(24,36,5, padding = 0, stride=2)
        self.conv3 = nn.Conv2d(36,48,5, padding = 0, stride=2)
        self.conv4 = nn.Conv2d(48,64,3, padding = 0)
        self.conv5 = nn.Conv2d(64,64,3, padding = 0)
#         self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*1*33, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
#         self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 64*1*33)
#         x = self.dropout(x)
        x = F.relu(self.fc1(x))
#         x = self.dropout(x)
        x = F.relu(self.fc2(x))
#         x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x 
    
model = Nvidia()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #load model
    # model = load_model(args.model)
    # checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    # model = checkpoint['net']
    # model = torch.load(args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
