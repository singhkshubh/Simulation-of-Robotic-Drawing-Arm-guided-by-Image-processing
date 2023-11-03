import cv2
#load image
img=cv2.imread(r"C:\Users\ksund\OneDrive\Desktop\trail\dog.png")
print(img.shape)
#converting to greyscale
img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show image#
cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#loading the data of img1- height, width,of the image
height = img1.shape[0]
width = img1.shape[1]
print(height)
print(width)
R=height
C=width
# resize the image
img2=cv2.resize(img1,(100,100))
#manipulating the pixel data
img2=img2<255*0.8
import tensorflow as tf
img2=tf.cast(img2,float).numpy()
print(img2)
# saving the text file
with open('dog.txt', 'w') as testfile:
    for row in img2:
        testfile.write(' '.join([str(img2) for img2 in row]) + '\n')
