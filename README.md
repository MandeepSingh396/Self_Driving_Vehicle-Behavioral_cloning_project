# Self_Driving_Vehicle-Behavioral_cloning_project
Using a Deep Learning model to train a vehicle to autonomously navigate using Unity simulator

## Simulator
the self driving car simulator can be downloaded from this link for windows : https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip
The self driving car simulator has two modes. One to generate the training data and the other for testing our trained model in autonomous mode.

![image](https://user-images.githubusercontent.com/81529956/178166826-dc75fda1-96fd-4cec-9f86-24168dd60cfb.png)

### Training mode: 
It is used to collect the training data for our deep learning model. You can normally drive the car around on the track as you do in any car video game i.e. usng the arrow keys.
But I would recommend using the mouse buttons for the steering input as doing such gives a continuous input rather than discreet inputs being given through keyboard buttons.
You can start the data collection by clicking on the recording button on the top right and start driving around the track. 5-6 laps around the track would give us enough data to train the deep learning model.

![image](https://user-images.githubusercontent.com/81529956/178166997-17ea8f28-c78a-412d-8f99-388e99ead962.png)
### Autonomous mode:
It is used to test how well our trained deep learning model is working. To use this we have to run driveN.py file along with the argument model_Nvidia.h5 (saved state dictionary of the trained model)
and run the simulator in autonomous model. You will see that the car will start driving around by itself as per the output from the trained model.

![image](https://user-images.githubusercontent.com/81529956/178167654-b1f10bac-c63c-4070-8301-2bf42b1f3509.png)

## Data Pre-processing
Before we give the data as input to our deep learning model we have to preprocess the data so that model is trained faster and efficiently. Below pre-processing techniques have been used:
1. Data Normalization: The RGB image pixels have value between 0-255. So, we normalize the pixel values to the range 0-1. Why we do normalization?You can read here :
https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
2. Data Augmentation: It should be ntoiced that the image data that we have collected from simulator is from an anti-clockwise track which has a lot of left turns and very few right turns.
This will hamper the performance of our model as it may not have enough data to learn the right handed turns. So, I have flipped all the images horizontally and added them to the original 
data such that we can train the model efficiently.
3. Image crop: The initial image size that we get from the simulator is 160x320. But all of the information in the image is not useful. Like, the top part of the image shows hills and lower portion of the images
shows the car bonet. We can crop these part of the images which will not effect our model in any way. The final image size after cropping is 65x320.

## Model Architecture
The CNN model used is Nvidia's self driving car model for autonomous vehicles. You can refer this blog post. https://developer.nvidia.com/blog/deep-learning-self-driving-cars/

![image](https://user-images.githubusercontent.com/81529956/178167839-a41c405a-7e9a-4ee2-ae5c-e5510fd5305a.png)

## Results
Once the model is trained and state dictionaries are loaded in model_Nvidia.h5 file, we can run the driveN.py file in terminal to run the vehicle autonomously in simulator.
Note: You can to run the simulator in autonomous mode first then only driveN.py file will run.

https://user-images.githubusercontent.com/81529956/178168054-7c8b0e77-5365-4075-b49e-9308e6a2745c.mp4


