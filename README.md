
# Aries_2y_proj

# UNET_EDGE_DETECTOR
The first step towards Implementing an Image Drawing Robotic arm is to get an edge map of the Image from which the coordinates can be extracted and can further be provided to the component for tracing the image on paper. And for this purpose, I have used Deep Learning models trained on BIPED Dataset.

## Dataset
It contains 250 outdoor images of 1280 x 720 pixels each. Experts in the computer vision field have carefully annotated these images. Hence redundancy has yet to be considered. In spite of that, all results have been cross-checked several times to correct possible mistakes or edges by just one subject. This dataset is publicly available as a benchmark for evaluating edge detection algorithms. 

## Model
The model's objective is to create an edge map of an input image. I have used a U-Net model because this work is very similar to semantic segmentation, as we classify each pixel into a category edge(1) or non-edge(0). So it is intuitive to get a concise feature representation of the image. Then reconstruct an edge map by upsampling these features, and for a better information flow, we have skip connections between components during downsampling and upsampling.
### U-NET
UNET is one of the most popular architectures used for image segmentation tasks because it is designed to efficiently capture local and global information in the input image while preserving fine-grained spatial information through skip connections. It consists of two parts: Encoder and Decoder.
The encoder is a series of convolutional layers that extract features from the input image. Hence, we downsample the image to a consistent and rich feature representation of the image.
The decoder then consists of a series of upsampling layers that gradually increase the spatial resolution of the feature maps. Each upsampling layer is then followed by a block consisting of two convolutional layers, which combine information from the corresponding layer in the encoder and the upsampled feature maps.
And in addition to this, the most crucial part of architecture is Skip Connections. These skip connections allow information from the encoder to be directly passed to corresponding layers in the decoder, preserving fine-grained spatial information and improving the accuracy of the segmentation.

### Pretrained VGG19 Encoder
But since the dataset on which the training is to be done contained only 200 images, training the unit from scratch led to overfitting. So to overcome that, I have used the VGG19 pre-trained model on ImageNet as the encoder part of unet for feature extraction. The intuition behind it was that the pre-trained model must have learned to get low-level and high-level features for recognizing image structure and context, which is quite a similar operation that our model should also do to extract a good feature representation of the image. Later that feature is used in upsampling and reconstructing the edge map image.

## Preprocessing and Image Augumentation
Fistly Image pixel were normalised which were in the range 0 to 255 to improve the performance of the model, particularly when using certain activation functions such as ReLU. This is because these activation functions are sensitive to the scale of the input values, and normalizing the input images helps to ensure that the activation functions are used in their most effective range. Moreover it also helps to avoid the problem of exploding or vanishing gradients during training.

The ground truth target is an image again consisting pixel value from 0 to 255 so this image is converted to edge map where each pixel is classified as 1(edge pixel) or 0 (non-edge pixel) so that training can be done to classify every pixels.

Since the dataset consist of very less number of images data augmentation can be one solution to increase the training dataset.But augumentation technique like roation will not be useful as in this case after rotation there are some empty pixel places wich are filled with last true pixel and since our edge map will have less number of edge pixel this error could hamper the performance of our model.Also changing contrast or brightness wil not be useful because for our downline usecase of this model there is hardly a chance where such image will be encountered so it's not worth doing.Left right flipping of image can be useful technique to make the model more robust and increase model performance.


## Loss Function
### Why Need of Custom Loss Function?
The most crucial part of this model is the loss function, as using BinaryCross Entropy loss fails here because the class is highly imbalanced, so the model learns to cheat and perform well in classifying non-edge pixels correctly. And hence assign all the pixels as non-edge because it reduces the loss function drastically, as these pixels have a high contribution to the loss.T his problem mainly occurs due to large number of non edge pixel i.e.. almost 97% of the total pixels so it's obvious that model will try to perform better on these pixels majorly but again that was not our objective.So there is a need of weighted cross entopy loss so that we can bring the contribution of loss due to classification of both edge and non edge pixel to same order so that our model tries to perform well in both cases.

Though we have some standard tensorflow fuction for weighted CrossEntropy Loss but those were not useful here because in those cases we were assigning a multiple to increase weight on true classification then here again the model was now getting biased to classify the edge pixels correctly and ended up assigning all pixel as edge pixel.Moreover, here the weight is a, new hyperparameter that needs to be tuned which is again a big problem and also the same weight of each class is used for every image that is also not correct because every image has different ratio of edge and non edge pixel.So catering all these problems I have used a dynamically weigheing loss function for each image.

### Custom Loss Function
**Loss Function:**  **-**(Beta)*(y_true)*(log(y_pred))-(1-Beta)*(1-y_true)*(log(1-y_pred))

**Beta** = (Total Number of Edge Pixel)/(Total Number of Pixel in Image)

By dynamically weighting the loss for every image, we bring the contribution of edge and nonedge classification to the same order so the model will learn to perform well on both classifications and not give a biased output.

## Metric
Binary Accuracy is not a good mertric here because the target is highly imbalenced so even if all pixel are assigned non-edge accuracy will we 97% which may seem great but it's not the actual case,hence this could be misleading.So here we use F1 score as our metric,which take into account not only the number of prediction errors that your model makes, but that also look at the type of errors that are made..

F1 score is harmonic mean of Precision and Recall value and is in range 0 to 1. 

Precision: Within everything that has been predicted as a positive, precision counts the percentage that is correct.
   
   Precision= Number of True Positives/(Number of True Positives + Number of False Positives)

Recall:Within everything that actually is positive, how many did the model succeed to find.
   
   Recall= Number of True Positives/(Number of True Positives + Number of False Negatives)

# Image to array conversion
The image,which is to be drawn,is first passed through edge detection machanism.After that an important task is to convert the image data to array format,which could br further used for determining the coordinates for movement of robotic arm .
for this purpose first the image is convertrd to grayscale using opencv library.
after converting to grayscale the image is resized to 100 * 100 .then the pixel data is manipulated using tenserflow library
if the pixel value is greater than some threshold value, it becomes 1 or 0.
by this way the image is converted to a 2d array of size 100 * 100.
this array is stored as a (.txt) file . 

# How to send final coordinates to Robotic Arm

In this segment, we detect the path in which the pen will move. First, we detect the portion which has to be drawn. Then, we make a visited array. Now, we can trace the path using the DFS (depth-first search) algorithm and a visited array. After that, we send the coordinate (output by DFS) to the robotic arm, through which it has to pass (or draw). We also send a special parameter with the coordinate, which tells the robotic arm to draw or lift the pen .
 
# Robotic_arm_Implementation

This repository provides a solution for controlling a robotic arm using an Arduino board to draw sketches or figures. The simulation is implemented using Matlab/Simulink.
## Hardware Implementation

Arduino Communication:-
 The Arduino board establishes real-time communication with a computer to receive coordinates for the pen position and placement. As a result, there is no need for external memory such as an SD card. The robotic arm can operate continuously without requiring disconnection for different uses.

Python Script:-
 'CoordinateTransfer.py' is included in this repository, which reads a set of coordinates from a text file and normalizes them based on the page size and position. The script establishes a serial communication channel between the computer and the Arduino board using a baud rate of 9600. Additionally, the script performs a verification check to ensure successful transmission of coordinates. If necessary, it can resend the coordinates.
## Simulation on MATLAB/Simulink
The verification of the sketch can be performed using the Robotics and Simscape Toolbox in Matlab. However, for this project, the inverse kinematics were hard-coded because the block provided by the robotics toolbox did not meet our requirements and was computationally expensive.

To work with this repository, please follow these steps:

1. Import the 'rbTree' file into the Matlab workspace to import the rigidbody tree. Import the text file containing the coordinates as sets of column vectors.
2. Run the 'trajectorybuilder.m' script to normalize the data to coordinates. Open the Simulink file 'inv_kin.slx'. 
3. Input the time vector and coordinate vectors into the signal builder within Simulink to define the desired path along the coordinates. Ensure that you choose an integer 'k' to normalize the time vector (T/k) in order to complete the simulation within a finite time. 
4. You can also view the robotic simulation within the Matlab editor. 

Please note that for this particular example, the coordinate space is set to 100x100 in the text file, representing the resized image.

## Results

The following sketches were obtained as a result from the program.
https://github.com/sundramkumar252636/aeries_2y_3dof_proj/assets/133499612/307b6963-42c2-4cc7-a6e0-24c7af264a1c
![image](https://github.com/sundramkumar252636/aeries_2y_3dof_proj/assets/133499612/9ff2b7a6-33cd-4bdb-95fd-b0323c05a441)
