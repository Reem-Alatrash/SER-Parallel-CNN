'''
@author: Reem Alatrash
@version: 1.0
@Python ver: 2.7
=======================

This script creates a convolutional neural network (CNN) with parallel convolution layers to classify the emotion embedded within speech signals (both scripted and improvised) into 1 of 4 emotions: happy, sad, angry, neutral.

The system takes as input pre-processed features extracted from the speech subset of the Interactive Emotional Dyadic Motion Capture (IEMOCAP) database. 
The system produces 2 files:
1. The model (parallel CNN), which is saved as 'parallel_cnn_BN.h5'.
2. The predictions for both validation/dev and test sets in tab-seperated files. Each prediction file contains the IDs audio signals and their corresponding predicted class.

Example:
MSP-IMPROV-S08A-F05-S-FM02	happy

Note(s): run in py 2.7 with the data extracted using py 2.7

'''

# magic function to enable nbAgg backend in Jupyter Notebook
# https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#magics-explained
%matplotlib nbAgg
%matplotlib nbAgg

'''
******* ********* *********
*******  imports  *********
******* ********* *********
'''
from sklearn import preprocessing
import xarray as xr
import numpy as np
import pickle

'''
******* ********* *********
******* variables *********
******* ********* *********
'''
# Step 1: Prepare data
input_dir = "./data/" 
#names = ['data_prosody.train', 'data_logMel.train', 'data_prosody.valid', 'data_logMel.valid', 'data_prosody.test', 
#         'data_logMel.test']
file_paths = [input_dir + "data_prosody.train", input_dir + "data_logMel.train", input_dir + "data_prosody.valid", 
              input_dir + "data_logMel.valid", input_dir + "data_prosody.test", input_dir + "data_logMel.test"]

logMel_train_ids = []
logMel_train_features = []
logMel_train_labels = []
logMel_valid_ids = []
logMel_valid_features = []
logMel_valid_labels = []
logMel_test_ids = []
logMel_test_features = []
logMel_test_labels = []
# have a fixed random seed for reproducibility
seed = 42
np.random.seed(seed)

'''
******* ********* *********
******* functions *********
******* ********* *********
'''
def unpickle_data(path):
    '''returns 3 lists: data Ids, features, labels'''    
    with open(path, 'rb') as f_in:
        ids = pickle.load(f_in)
        features = pickle.load(f_in)
        labels = pickle.load(f_in)
        #print(len(ids), len(features), len(labels))
        #print("{0}\n{1}\n{2}".format(ids[10],features[10],labels[10]))
        return ids, features, labels

# 1.1 unpickle files (load data)
logMel_file_paths = list(x for x in file_paths if "logMel" in x)
#print(logMel_file_paths)
logMel_train_ids,logMel_train_features,logMel_train_labels = unpickle_data(logMel_file_paths[0])
logMel_valid_ids,logMel_valid_features,logMel_valid_labels = unpickle_data(logMel_file_paths[1])
logMel_test_ids,logMel_test_features,logMel_test_labels = unpickle_data(logMel_file_paths[2])

#print(logMel_valid_labels)

###################################################################
#### WARNING: Don't run this section until the model is stable ####
###################################################################
# 1.2 increase training set size using some examples from the validation set

# 1.2.0 get distribution of validation set 
# we would like to perserve the distribution of data
import collections
# create a dictionary with the frequencies of the 4 labels
counts_valid = collections.Counter(logMel_valid_labels)
print counts_valid
# create a dictionary with the percentages of the labels
ratio_valid = {class_label : counts_valid[class_label]*1.0 / len(logMel_valid_labels) 
               for class_label in counts_valid.keys()}
print ratio_valid

# 1.2.2 split the combined data according to the specified ratio
# params
old_train_size = len(logMel_train_ids)
combined_size = old_train_size + len(logMel_valid_ids)
split_ratio = 0.7 # Ex: 0.8 of combined data => 80% training, 20% dev
new_train_size = int(combined_size * split_ratio)
total_examples_to_extract = new_train_size - old_train_size

print("old train size ", old_train_size)
print("new train size ", new_train_size)
print("examples to extract ", total_examples_to_extract)

from math import modf, floor
# 1.2.3 calculate no. of examples to extract for each class/label
# split values into whole and decimal parts in order to perform Largest Remainder Method
# https://stackoverflow.com/questions/13483430/how-to-make-rounded-percentages-add-up-to-100
ratio_examples_modf = {class_label : modf(ratio_valid[class_label]* total_examples_to_extract)
                  for class_label in ratio_valid.keys()}

# sort dictionary by the value of the decimals
ordered_by_decimals = collections.OrderedDict(sorted(ratio_examples_modf.items(), 
                                                     key=lambda t: t[1][0], reverse=True))
print(ratio_examples_modf)
print(ordered_by_decimals)

# ratio_examples =  {'angry': 109.0, 'sad': 121.0, 'neutral': 477.0, 'happy': 362.0}
ratio_examples = {class_label : floor(ratio_valid[class_label]* total_examples_to_extract)
                  for class_label in ratio_valid.keys()}
print(ratio_examples)
# make sure the number of examples from ratios doesn't exceed the split ratio
current_total = sum(ratio_examples.values()) 

diff_ex = total_examples_to_extract - current_total
if diff_ex > 0:
    # add 1 to the top diff items | e.g. diff = 3, add 1 to the top 3 items
    for label in ordered_by_decimals.keys()[:int(diff_ex)]:
        ratio_examples[label] += 1
        
print(ratio_examples)        


# 1.2.4 Add examples to Training data set

# ordered of the labels in the validation set
label_order = ["angry", "happy", "sad", "neutral"]
subset_start = 0
# get the start and end ID for each label subset that needs to be extracted
# {'angry': (0, 109), 'sad': (1718, 1839), 'neutral': (2160, 2637), 'happy': (396, 758)}
valid_subsets_indixes = {}

for label in label_order:
    # add number of examples to be extracted the start index to get subset end
    subset_end = subset_start + int(ratio_examples[label])
    # set start and end of each label subset
    valid_subsets_indixes[label] = (subset_start,subset_end)
    # compute start index for next subset/label
    subset_start += int(counts_valid[label])

print(valid_subsets_indixes)

# initialize the extended sets with the old train set and old valid set
logmel_ext_train_ids = logMel_train_ids
logmel_ext_train_features = logMel_train_features
logmel_ext_train_labels = logMel_train_labels

logmel_ext_valid_ids = logMel_valid_ids
logmel_ext_valid_features = logMel_valid_features
logmel_ext_valid_labels = logMel_valid_labels


# change the data sets by extending training set and shrinking validation set
for label in label_order:
    label_range = valid_subsets_indixes[label]

    # new train set: append samples from old validation set
    logmel_ext_train_ids = np.append(logmel_ext_train_ids,logMel_valid_ids[label_range[0]:label_range[1]],axis=0)
    logmel_ext_train_features = np.append(logmel_ext_train_features,
                                          logMel_valid_features[label_range[0]:label_range[1]],axis=0)
    logmel_ext_train_labels = np.append(logmel_ext_train_labels,
                                        logMel_valid_labels[label_range[0]:label_range[1]],axis=0)     

# new validation set: delete samples from old validation set
for label in label_order[::-1]:
    label_range = valid_subsets_indixes[label]
    logmel_ext_valid_ids = np.delete(logmel_ext_valid_ids, range(label_range[0],label_range[1]),axis=0)
    logmel_ext_valid_features = np.delete(logmel_ext_valid_features, range(label_range[0],label_range[1]),axis=0)
    logmel_ext_valid_labels = np.delete(logmel_ext_valid_labels, range(label_range[0],label_range[1]),axis=0)  

# sanity check    
print(logmel_ext_valid_ids[0])
print(logMel_valid_ids[109])

# override the old variables with the new data
logMel_train_ids = logmel_ext_train_ids
logMel_train_features = logmel_ext_train_features
logMel_train_labels = logmel_ext_train_labels

logMel_valid_ids = logmel_ext_valid_ids
logMel_valid_features = logmel_ext_valid_features
logMel_valid_labels = logmel_ext_valid_labels

#1.3 Normalize data
train_scaler = preprocessing.Normalizer(norm='l2', copy=False).fit(logMel_train_features[0,:,:])
valid_scaler = preprocessing.Normalizer(norm='l2', copy=False).fit(logMel_valid_features[0,:,:])
test_scaler = preprocessing.Normalizer(norm='l2', copy=False).fit(logMel_test_features[0,:,:])
train_scaler.transform(logMel_train_features[0,:,:])
valid_scaler.transform(logMel_valid_features[0,:,:])
test_scaler.transform(logMel_test_features[0,:,:])
#print(logMel_train_features[0])

# 1.4 encode the string labels into numeric values    
le = preprocessing.LabelEncoder()
le.fit(["angry", "happy", "sad", "neutral"])
logMel_train_encoded_labels = le.transform(logMel_train_labels)
logMel_valid_encoded_labels = le.transform(logMel_valid_labels)

# to decode these labels use the inverse_transform function. Example: 
#results = list(le.inverse_transform([2, 2, 1]))

# 1.4.1 One hot encoding for the target vectors
from keras.utils import np_utils
y_logMel_train_cat = np_utils.to_categorical(logMel_train_encoded_labels)
y_logMel_valid_cat = np_utils.to_categorical(logMel_valid_encoded_labels)


# 1.5 reshape data into 4D matrix (CNN takes 4D) by adding no. of input channels
# for tensorflow no. of input channels is the last dimension (Theano has it as first)
# params
input_height = 750 # height = no. of rows/segments (750)
input_width = 26 # this is the no. of cols (26 in logmel file)
input_channels = 1
#X_train_4D = np.expand_dims(logMel_train_features, axis=-1)
X_logMel_train_4D = logMel_train_features.reshape(logMel_train_features.shape[0], input_height, 
                                                  input_width , input_channels).astype('float32')
print(X_logMel_train_4D.shape)
X_logMel_valid_4D = logMel_valid_features.reshape(logMel_valid_features.shape[0], input_height, 
                                                  input_width , input_channels).astype('float32')
X_logMel_test_4D = logMel_test_features.reshape(logMel_test_features.shape[0], input_height, 
                                                input_width , input_channels).astype('float32')

# Step 2: Define model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.merge import Concatenate
from keras import optimizers, metrics

# Paramters
filters_l1 = 100
filter_heights = [5,10]
filter_width = input_width
pool_height = [30,26]
pool_width = 1
pool_stride = 3
filter_stride = 3
num_classes = 4
loss_func = 'categorical_crossentropy' #'mse' # 'binary_crossentropy' 
#MSE is used for regression mostly (see ex3 and https://www.youtube.com/watch?v=IVVVjBSk9N0)
optimizer_func = 'adam'
conv_actv = 'relu'
dropout_rate = 0.5 #0.25 gave best results
img_path='network_image.png'
metric =  [metrics.categorical_accuracy]#['accuracy']

# 2.0 begin model design
# Parallel layers code from Keras.pdf and here:
# https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras

# 2.1 input shape defined for our variant filter-width layer 
model_input = Input(shape=X_logMel_train_4D.shape[1:])
conv_blocks = []
for i in range(0,2):
    # add a conv layer with the specified width 
    conv = Conv2D(filters=filters_l1, kernel_size=(filter_heights[i], filter_width), 
                  activation=conv_actv,strides=filter_stride)(model_input)
    print("conv_{} shape: {}".format(i, conv.shape))
    # add pooling 
    pooled = MaxPooling2D(pool_size=(pool_height[i], pool_width),strides =pool_stride)(conv)
    print("pooled_{} shape: {}".format(i, pooled.shape))
    # flatten results
    #pooled_flat = Flatten()(pooled)
    conv_blocks.append(pooled)#(pooled_flat)
if len(conv_blocks) > 1: # here we merge
    cp_all = Concatenate(axis=-1)(conv_blocks)
else:
    cp_all = conv_blocks[0]
    
# 2.2 add dense layer to handle the output of the previous layer
print("concatinated tensor shape: {}".format(cp_all.shape))

# Dense documentations flattens the input tensor by default if its rank is > 2 
# https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L787
den = Dense(100, activation='relu')(cp_all)
print("dense_1 tensor shape: {}".format(den.shape))

# The extra flattening below yields better results.
# Notes: 
# 1) the number of parameters in the sub model goes down from 1,509,300 to 59,300. However, for the softamx layer 
# the no. of params increases from 404 to 29,604 
# 2) the model seems to predict anger and sadness more accuaretly with this change.
model_output = Flatten()(den)

#print(model_output.shape)

# 2.3 define sub model (our convoluaitonal layer with various filter sizes)
sub_model = Model(model_input, model_output)

# 2.4 define overall model
model = Sequential()
# 2.5 first convolutional layer
model.add(sub_model)

BatchNormalization(momentum=0.99)
# 2.6 add dropout
model.add(Dropout(dropout_rate))
BatchNormalization(momentum=0.99)

# 2.7 add dense (fully connected) layer for the softmax
model.add(Dense(num_classes, activation='softmax'))  
# note: # params >> # training instances is bad and you should reconsider your design
model.summary()
# 2.8 save structure of model
from keras.utils import plot_model
plot_model(model, to_file=img_path)

#2.9 compile model
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=True)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=False)
optimizer_func = adam #sgd 
model.compile(loss=loss_func, optimizer=optimizer_func, metrics=metric)          

# Step 3: Train model

# define path to save best model
model_path = './parallel_cnn_BN.h5'

# 3.0 prepare callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

## use earlstopping to avoid overfitting with epochs
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=15, mode='max', verbose=1)
# create a checkpoint to save the best model based on validation loss
acc_chkpoint = ModelCheckpoint(model_path, monitor='val_categorical_accuracy', 
                                save_best_only=True, mode='max',verbose=0)
callbacks = [acc_chkpoint,early_stop]
# Paramters
total_epochs = 50
batch_s = 32
number_training_samples = X_logMel_train_4D.shape[1]

history = model.fit(X_logMel_train_4D, y_logMel_train_cat, validation_data=(X_logMel_valid_4D, y_logMel_valid_cat),
                    epochs=total_epochs, batch_size=batch_s, verbose=2, shuffle=True,callbacks = callbacks)
#history = model.fit(X_logMel_train_4D, y_logMel_train_cat, validation_data=(X_logMel_train_4D, y_logMel_train_cat), 
#                    epochs=total_epochs, batch_size=batch_s, verbose=2, shuffle=True)
        
# 3.1 evaluate model on dev set
score = model.evaluate(X_logMel_valid_4D, y_logMel_valid_cat, verbose=0)
#score = model.evaluate(X_logMel_train_4D, y_logMel_train_cat, verbose=0)

print(model_output.shape)

#3.2 get score
print("Accuracy on dev set:{0}%".format(score[1]*100))


import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Step 4: Use model to predict
# 4.1 Get Probability distribution over the classes shape:(elements, classes)
prediction = model.predict(X_logMel_test_4D)

print("Prediction shape:{0}".format(prediction.shape))

print prediction[0]

# 4.2 Get the predictions list
dev_predicted_classes = model.predict(X_logMel_valid_4D).argmax(axis=1)
predicted_classes = model.predict(X_logMel_test_4D).argmax(axis=1)
print("Prediction shape:{0}".format(predicted_classes.shape))
if len(logMel_test_ids) == len(predicted_classes):
    print("ids and classes lengths match")
print("{0}\t{1}".format(logMel_test_ids[0],predicted_classes[0]))

# 4.3 Map predictions to ids and save to file
results = list(le.inverse_transform(predicted_classes))
dev_results = list(le.inverse_transform(predicted_classes))
#4.4 save results to file

with open("predictions_parallel_CNN_feb_8_dev.txt", 'wb+') as f_out:
    for idx in range(0,len(logMel_valid_ids)):
        f_out.write("{0}\t{1}\n".format(logMel_valid_ids[idx],dev_results[idx]))
print("done") 

with open("predictions_parallel_CNN_feb_8.txt", 'wb+') as f_out:
    for idx in range(0,len(logMel_test_ids)):
        f_out.write("{0}\t{1}\n".format(logMel_test_ids[idx],results[idx]))
print("done") 

# Step 5: Manually check the results of the dev prediction
# params
correct = 0

counts_valid = collections.Counter(logMel_valid_labels)
print(counts_valid)

c_pred_ang = 0
c_pred_hap = 0
c_pred_sad = 0
c_pred_neu = 0

# get count of correct label w.r.t. label
def count_labels_by_cat(correct_label):
    #print(correct_label)
    global c_pred_ang
    global c_pred_hap
    global c_pred_sad
    global c_pred_neu
    if correct_label == "happy":
        c_pred_hap += 1
    elif correct_label == "angry":
        c_pred_ang += 1
    elif correct_label == "sad":
        c_pred_sad += 1
    else:
        c_pred_neu += 1

for idx in range(0,len(logMel_valid_ids)):
    if dev_results[idx] == logMel_valid_labels[idx]:
        correct += 1
        count_labels_by_cat(dev_results[idx])
print("--------------------")
print "Correct Validation predictions:", correct
print "Valdiation Accuracy:", (correct*1.0)/len(logMel_valid_labels)
print("--------------------")
print "Angry acc:", (c_pred_ang*1.0)/counts_valid['angry']
print "Happy acc:", (c_pred_hap*1.0)/counts_valid['happy']
print "Sad acc:", (c_pred_sad*1.0)/counts_valid['sad']
print "Neutral acc:", (c_pred_neu*1.0)/counts_valid['neutral']
