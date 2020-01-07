from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import sys

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

np.set_printoptions(threshold=sys.maxsize)

### Step 1 : find face + Step 2 : crop around face
##################################################
def auto_crop_image(image):
    if image is not None:
        im = image.copy()
        faceCascade = cv2.CascadeClassifier('/home/walsam/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        if len(faces) > 0:
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                roi_color = image[y:y+h, x:x+w]
                img_item = "anchor.png"
                cv2.imwrite(img_item,cv2.resize(roi_color,(96,96)))
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)        
            (x, y, w, h) = faces[0]
            
            center_x = x+w/2
            center_y = y+h/2
            height, width, channels = im.shape
            b_dim = min(max(w,h)*1.2,width, height)
            box = [center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2]
            box = [int(x) for x in box]
            
            # Crop Image
            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                #for (x, y, w, h) in faces:
                #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)        
                #    cv2.imwrite('only_face.jpg')  #--- Here is where the increment variable is placed. It will be incremented for every face and thus saving every face that gets detected.
                crpim = im[box[1]:box[3],box[0]:box[2]]
                crpim = cv2.resize(crpim, (224,224), interpolation = cv2.INTER_AREA)
                print("Found {0} faces!".format(len(faces)))
                return crpim, image, (x, y, w, h)
    return None, image, (0,0,0,0)

# triplet_loss

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist-neg_dist+alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss
# GRADED FUNCTION: who_is_it

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.5:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print("Total Params:", FRmodel.count_params())
    with tf.compat.v1.Session() as test:
        tf.compat.v1.set_random_seed(1)
        y_true = (None, None, None)
        y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
        loss = triplet_loss(y_true, y_pred)
    
        print("loss = " + str(loss.eval()))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    #Loading database
    database = {}
    database["walid 1"] = img_to_encoding("images/walsam.png", FRmodel)
    database["walid 2"] = img_to_encoding("images/walid.png", FRmodel)
    database["hamza"] = img_to_encoding("images/hamza.png", FRmodel)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        imgcrop,frame, (x, y, w, h) = auto_crop_image(frame)

        cv2.imshow('frame',frame)
        who_is_it("anchor.png", database, FRmodel)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()