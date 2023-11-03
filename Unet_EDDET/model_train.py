# Importing required libraries
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
import PIL.Image
import os
from PIL import Image, ImageOps
import cv2
from tensorflow.keras import layers

# Hyper Parameters
batch=32
im_h=512
im_w=512

train_x=tf.keras.utils.image_dataset_from_directory(r"Dataset\train\train_x",labels= None,label_mode='int',class_names="train_x", color_mode='rgb',batch_size=batch,image_size=(im_h, im_w),shuffle=False)
train_y=tf.keras.utils.image_dataset_from_directory(r"Dataset\train\train_y",labels= None,label_mode='int',class_names="train_y", color_mode='grayscale',batch_size=batch,image_size=(im_h, im_w),shuffle=False)

X=[]

for img in train_x:
    for i in img :
     X.append(i)
X=tf.convert_to_tensor(X)


Y=[]

for img in train_y:
    for i in img :
     Y.append(i[:,:,0])
     
Y=tf.convert_to_tensor(Y)
Y=tf.expand_dims(Y,-1)

dataset=tf.data.Dataset.from_tensor_slices({"pixel_values":X, "labels":Y})

def preprocess(data):
  image = data["pixel_values"]
  label=data["labels"]
  # NORMALISATION
  image = (image / 255.0)
  label= (label/255.0)
  # converting edge map to 0 and 1 mask
  label=(label>0.5)
  label=tf.cast(label,dtype=float)
  return image,label


def augument(data):
  image,label= preprocess(data)
  if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       image = tf.image.flip_left_right(image)
       label = tf.image.flip_left_right(label)

  return image, label


AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    dataset
    .map(augument, num_parallel_calls=AUTOTUNE)
    .batch(batch)
    
)

val_x=tf.keras.utils.image_dataset_from_directory(r".\Dataset\val_x",labels= None,label_mode='int',class_names="val_x", color_mode='rgb',batch_size=batch,image_size=(im_h, im_w),shuffle=False)
val_y=tf.keras.utils.image_dataset_from_directory(r".\Dataset\val_y",labels= None,label_mode='int',class_names="val_y", color_mode='grayscale',batch_size=batch,image_size=(im_h, im_w),shuffle=False)


val_X=[]

for img in val_x:
    for i in img :
     val_X.append(i)
val_X=tf.convert_to_tensor(val_X)


val_Y=[]

for img in val_y:
    for i in img :
     val_Y.append(i[:,:,0])
     
val_Y=tf.convert_to_tensor(val_Y)
val_Y=tf.expand_dims(val_Y,-1)

val_dataset=tf.data.Dataset.from_tensor_slices({"pixel_values":val_X, "labels":val_Y})



AUTOTUNE = tf.data.AUTOTUNE
val_ds = (
    val_dataset
    .map(preprocess, num_parallel_calls=AUTOTUNE)  
    .batch(30)
)


from tensorflow.keras.applications.vgg19 import VGG19
base_model=tf.keras.applications.vgg19.VGG19(include_top=False,input_shape=(512,512,3),weights='imagenet')

base_model= tf.keras.Model([base_model.input],[base_model.get_layer('block4_conv2').output])

def conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   x = tf.keras.layers.BatchNormalization()(x)  
   return x

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D with ReLU activation
   x = conv_block(x, n_filters)
   return x


  # UNET MODEL WITH VGG19 AS ENCODER

set_trainable = False
for layer in base_model.layers:
  if layer.name in ['block1_conv1']:
        set_trainable = True

inputs= base_model.input
last_layer = base_model.output
bottle_neck=conv_block(last_layer,512)
u1=upsample_block(bottle_neck,base_model.get_layer("block3_conv4").output,256)
u2=upsample_block(u1,base_model.get_layer("block2_conv2").output,128)
u3=upsample_block(u2,base_model.get_layer("block1_conv2").output,64)
outputs= layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u3)
unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
unet_model.summary()


import tensorflow.keras.backend as K
class cross_entropy_balanced(tf.keras.losses.Loss):
  def __init__(self, name="custom_mse"):
        super().__init__(name=name)


  def call(self,y_true, y_pred):
    

    y_pred   = tf.clip_by_value(y_pred,K.epsilon(), 1 - K.epsilon())

    y_true = tf.cast(y_true, tf.float32) 
    img_cost=0
    
    
    count_neg = tf.math.reduce_sum(1. - y_true,[1,2,3])
    count_pos = tf.math.reduce_sum(y_true,[1,2,3])
    beta=count_pos / (count_neg + count_pos)
    pr_1=-tf.multiply(y_true,tf.math.log(y_pred))
    pr_0=-tf.multiply((1.-y_true),tf.math.log(1.-y_pred))
    img_cost=(beta*tf.math.reduce_sum(pr_0,[1,2,3]))+((1-beta)*tf.math.reduce_sum(pr_1,[1,2,3]))
    img_cost=tf.math.reduce_sum(img_cost)
     

    return img_cost

unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss=cross_entropy_balanced(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=(0.68)),tf.keras.metrics.Precision(thresholds=(0.68)),tf.keras.metrics.Recall(thresholds=(0.68))])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
       
        # The saved model name will include the current epoch.
        filepath="./checkpoints/my_checkpoint_unet",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
]

NUM_EPOCHS = 15
history=unet_model.fit(train_ds,epochs=NUM_EPOCHS,callbacks=callbacks,validation_data=val_ds,shuffle=False)

