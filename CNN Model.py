#!/usr/bin/env python
# coding: utf-8

# In[81]:


import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers


# In[82]:


from keras.utils import to_categorical
import idx2numpy
import skimage as sk
import scipy
import seaborn as sns


# In[2]:


# Reading files 

dataset1 = 'train-images-idx3-ubyte'
dataset2 = 'train-labels-idx1-ubyte'
dataset3 = 't10k-images-idx3-ubyte'
dataset4 = 't10k-labels-idx1-ubyte'

# Convert files into numpy array
train_X = idx2numpy.convert_from_file(dataset1)
train_Y = idx2numpy.convert_from_file(dataset2)
test_X = idx2numpy.convert_from_file(dataset3)
test_Y = idx2numpy.convert_from_file(dataset4)


# In[3]:


print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)


# In[4]:


# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[5]:


# Functions for various type of augmentation within the dataset

def Rotation(img):
    return sk.transform.rotate(img,30)

def Color_invert(img):
    return sk.util.invert(img)

def Flip(img):
    return np.flip(img)

def Random_Noise(img):
    return sk.util.random_noise(img)

def Contrast_Change(img):
    return sk.exposure.rescale_intensity(img, in_range=(10,120))

def Gamma_Correction(img,gamma=0.4):
    return sk.exposure.adjust_gamma(img,gamma)

def Sigmoid_Correction(img):
    return sk.exposure.adjust_sigmoid(img)


# In[6]:


# Function takes image array and integer as input and returns the corresponding augmented image 

def Augmentation(img, n):
    if(n==0):
        return Rotation(img)
    elif(n==1):
        return Color_invert(img)
    elif(n==2):
            return Flip(img)
    elif(n==3):
            return Random_Noise(img)
    elif(n==4):
            return Contrast_Change(img)
    elif(n==5):
            return Gamma_Correction(img)
    elif(n==6):
            return Sigmoid_Correction(img)
    else:
        return img


# In[7]:


Title = ["Rotation","Color_invert","Flip","Random_Noise","Contrast_Change","Gamma_Correction","Sigmoid_Correction"]


# In[8]:


# Visualize augmented images 

plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.imshow(train_X[0,:,:])
plt.xticks([])
plt.yticks([])
plt.title("Original")

for i in range(7):
    plt.subplot(2,4,i+2)
    imx = Augmentation(train_X[0,:,:],i)
    plt.imshow(imx)
    plt.xticks([])
    plt.yticks([])
    plt.title(Title[i])


# In[ ]:





# In[ ]:


################################################################################################


# In[58]:


((train_data, train_labels),
 (test_data, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


# In[59]:


target_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}


# In[60]:


print(train_data.shape)
print(test_data.shape)


# In[61]:


plt.figure(figsize=(10,10))
for i in range(0,20):
    plt.subplot(5,5, i+1)
    plt.imshow(train_data[i] )
    plt.title( target_dict[(train_labels[i]) ])
    plt.xticks([])
    plt.yticks([])


# In[62]:


train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)

test_data = test_data/np.float32(255)
test_labels = test_labels.astype(np.int32)


# In[63]:


def cnn_model(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    #dropout_1 = tf.layers.dropout(inputs=pool2, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv3 = tf.layers.conv2d(
          inputs= pool2,
          filters=128,
          kernel_size=[5, 5],
        
          padding="same",
          activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    #dropout_2 = tf.layers.dropout(inputs=pool3, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
       
    flatten_1= tf.reshape(pool3, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024,activation=tf.nn.relu)
    
    #dropout= tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    output_layer = tf.layers.dense(inputs= dense, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[64]:


fashion_classifier = tf.estimator.Estimator(model_fn = cnn_model)


# In[65]:


# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier.train(input_fn=train_input_fn, steps=500)


# In[67]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

test_results = fashion_classifier.evaluate(input_fn=test_input_fn)
print(test_results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


################################################## With dropout #########################################################


# In[71]:


def cnn_model_dropout(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #print(pool1.shape())
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
        
          padding="same",
        
          activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #print(pool2.shape())
    dropout_1 = tf.layers.dropout(inputs=pool2, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv3 = tf.layers.conv2d(
          inputs= dropout_1,
          filters=128,
          kernel_size=[5, 5],
        
          padding="same",
          activation=tf.nn.relu)
    
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    #print(pool3.shape())
    dropout_2 = tf.layers.dropout(inputs=pool3, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
       
    flatten_1= tf.reshape(dropout_2, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024,activation=tf.nn.relu)
    
    dropout= tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    output_layer = tf.layers.dense(inputs= dropout, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[72]:


fashion_classifier1 = tf.estimator.Estimator(model_fn = cnn_model_dropout)


# In[73]:


# Train the model
train_input_fn1 = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier1.train(input_fn=train_input_fn1, steps=100)


# In[76]:


test_input_fn1 = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

test_results1 = fashion_classifier1.evaluate(input_fn = test_input_fn1)
print(test_results1)


# In[75]:


################################################ Batch normalization ######################################


# In[ ]:





# In[ ]:





# In[77]:


def cnn_model_batchnorm(features, labels, mode):
    #Reshapinng the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
     # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same"
        )
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    
    pool1 = tf.layers.max_pooling2d(inputs= conv1, pool_size=[2, 2], strides=2)
    #print(pool1.shape())
    
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
          inputs = pool1,
          filters = 64,
          kernel_size=[5, 5],
          padding="same"
          )
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)
    
    pool2 = tf.layers.max_pooling2d(inputs= conv2, pool_size=[2, 2], strides=2)
    
    
    #dropout_1 = tf.layers.dropout(inputs=pool2, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
    
    # Convolutional Layer #2 and Pooling Layer #2
    conv3 = tf.layers.conv2d(
          inputs = pool2,
          filters = 128,
          kernel_size = [5, 5],
        
          padding = "same")
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.relu(conv3)
    pool3 = tf.layers.max_pooling2d(inputs= conv3, pool_size=[2, 2], strides=2)
    
    
    
    #dropout_2 = tf.layers.dropout(inputs=pool3, rate=0.25,training=mode == tf.estimator.ModeKeys.TRAIN )
       
    flatten_1= tf.reshape(pool3, [-1, 3*3*128])
    
    dense = tf.layers.dense(inputs= flatten_1,units=1024,activation=tf.nn.relu)
    
    #dropout= tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    output_layer = tf.layers.dense(inputs= dense, units=10)
    predictions={
    "classes":tf.argmax(input=output_layer, axis=1),
    "probabilities":tf.nn.softmax(output_layer,name='softmax_tensor')
    }
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss= tf.losses.sparse_softmax_cross_entropy(labels=labels, logits= output_layer, scope='loss')
    
    if mode== tf.estimator.ModeKeys.TRAIN:
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
        train_op= optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_op )
    
    eval_metrics_op={ "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_op)


# In[78]:


fashion_classifier2 = tf.estimator.Estimator(model_fn = cnn_model_batchnorm)


# In[79]:


# Train the model
train_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)

fashion_classifier2.train(input_fn=train_input_fn2, steps=1500)


# In[80]:


test_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_data},
    y=test_labels,
    num_epochs=1,
    shuffle=False)

test_results2 = fashion_classifier2.evaluate(input_fn=test_input_fn2)
print(test_results2)


# In[ ]:





# In[ ]:


######################################################## Early Stopping #################################################


# In[3]:


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


# In[4]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# In[5]:


# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


# In[6]:


model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()


# In[7]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# In[24]:


from tensorflow.keras.callbacks import EarlyStopping

earlyStop=EarlyStopping(monitor="val_loss",verbose=1,mode='min',patience=3)
var1 = model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[earlyStop])


# In[25]:


score = model.evaluate(x_valid, y_valid, verbose=0)


# In[26]:


#print loss and accuracy on test dataset
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[27]:


score_test = model.evaluate(x_test, y_test, verbose=0)


# In[28]:


#print loss and accuracy on test dataset
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1])


# In[40]:


# list all data in history
print(var1.history.keys())
# summarize history for accuracy
plt.plot(var1.history['acc'])
plt.plot(var1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(var1.history['loss'])
plt.plot(var1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[36]:


########################################ReduceLROnPlateau#######################################


# In[29]:


from tensorflow.keras.callbacks import ReduceLROnPlateau


# In[30]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')
var2 = model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid),
         callbacks=[reduce_lr])


# In[ ]:





# In[17]:


score1 = model.evaluate(x_valid, y_valid, verbose=0)


# In[18]:


#print loss and accuracy
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])


# In[20]:


score3 = model.evaluate(x_test, y_test, verbose=0)


# In[21]:


#print loss and accuracy on test dataset
print('Test loss:', score3[0])
print('Test accuracy:', score3[1])


# In[41]:


# list all data in history
print(var1.history.keys())
# summarize history for accuracy
plt.plot(var2.history['acc'])
plt.plot(var2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(var2.history['loss'])
plt.plot(var2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


########################################################################################################################

