import sys, os, multiprocessing, csv
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import cv2

IMSIZE = [64, 128]
#Read from CSV to get keys for images
def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:3:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header

def relabel(labeled_list):
    # labled_list_int = [int(x) for x in labeled_list]
    oldLabels = np.array(np.unique(labeled_list)[:])
    # if any(int(oldLabels) <= 12):
    #     #this algorithm will fail in the case where any label is less than 12
    #     #this is due to reusing the same list after relabeling
    #     print('relabling failed')

    idx = 0
    for lable in oldLabels:
        labeled_list = list(map(lambda x: x.replace(lable, str(idx)), labeled_list))
        idx = idx+1
    return labeled_list
#read images from list of keys 
def readImages(imageKey):
    folder = './data/images/'
    hogs = np.ndarray([len(imageKey), 3780])
    imIdx = 0
    for key in imageKey:
        filename = folder + key + '.jpg'
        rawim = cv2.imread(filename,0)
        imresize = cv2.resize(rawim,IMSIZE)
        im_hog = extractHog(imresize)
        hogs[imIdx,:] = im_hog
        imIdx = imIdx+1
    return hogs

def extractHog(img):
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    hog_vec = hog.compute(img)
    return hog_vec

def testTunings(numNodes, activation, batch_size, epochs, droupout):
    model = Sequential()
    model.add(Dense(numNodes, activation='relu', input_shape=(dim_data,)))
    model.add(Dropout(droupout))

    model.add(Dense(numNodes, activation='relu'))
    model.add(Dropout(droupout))

    model.add(Dense(classes_num_train, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, \
                    validation_data=(test_data, test_labels_one_hot))
    [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
    return test_acc, history

data_file = './data/google-data/df_final.csv'
all_data = ParseData(data_file)
all_data = all_data
all_keys = [x[0] for x in all_data]
all_labels = [x[1] for x in all_data]
#lables need to be relabled because the 'to_catagorical' function needs the labels
# to be labled numerically in order (i think)
all_labels = relabel(all_labels)

trainamount = 8000
train_keys = np.array(all_keys[0:trainamount])
train_hogs = readImages(train_keys)
train_labels = np.array(all_labels[0:trainamount])

test_keys = np.array(all_keys[trainamount-1:-1])
test_hogs = readImages(test_keys )
test_labels = np.array(all_labels[trainamount-1:-1])



print('Training data shape : ', train_keys.shape, train_labels.shape)

print('Testing data shape : ', test_keys.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes_train = np.unique(train_labels)
classes_test = np.unique(test_labels)
classes_num_train = len(classes_train)
classes_num_test = len(classes_test)
print('Total number of train outputs : ', classes_num_train)
print('Total number of test outputs : ', classes_num_test)
print('Output classes : ', classes_train)
print('Output classes : ', classes_test)

# plt.figure(figsize=[10,5])

# Display the first image in training data
# plt.subplot(121)
# # singleTrain = train_images[0]
# # plt.imshow(singleTrain, cmap='gray')
# plt.imshow(train_images[0,:,:], cmap='gray')
# plt.title("Ground Truth : {}".format(train_labels[0]))

# Display the first image in testing data
# plt.subplot(122)
# plt.imshow(test_images[0,:,:], cmap='gray')
# plt.title("Ground Truth : {}".format(test_labels[0]))

dim_data = np.prod(train_hogs.shape[1:])
train_data = train_hogs.reshape(train_hogs.shape[0], dim_data)
train_data = train_data.astype('float32')

dim_data = np.prod(test_hogs.shape[1:])
test_data = test_hogs.reshape(test_hogs.shape[0], dim_data)
test_data = test_data.astype('float32')

train_labels_one_hot = to_categorical(train_labels, num_classes= classes_num_train)
test_labels_one_hot = to_categorical(test_labels, num_classes = classes_num_train)

num_nodes_list = [1300,1500,1800,2000]
batches_sz_list = [100, 200, 400]
epochs_list = [20, 40]
dropout_list = [0.1, 0.2, 0.3]

tunings = []
for epochs in epochs_list:
    for batch_sz in batches_sz_list:
        for num_nodes in num_nodes_list:
            for dropout in dropout_list:
                [test_acc, hist] = testTunings(num_nodes, [],batch_sz, epochs, dropout)
                tuning = {'acc':test_acc, 'nodes': num_nodes, 'batches': batch_sz, 'epochs':epochs, 'dropout':dropout, 'hist':hist}
                tunings.append(tuning)

# model = Sequential()
# model.add(Dense(1800, activation='relu', input_shape=(dim_data,)))
# # model.add(Dropout(0.3))

# model.add(Dense(1800, activation='relu'))
# # model.add(Dropout(0.3))

# model.add(Dense(classes_num_train, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(train_data, train_labels_one_hot, batch_size=100, epochs=30, verbose=1, \
#                    validation_data=(test_data, test_labels_one_hot))

# [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
# plt.show()

#Plot the Accuracy Curves
plt.figure(figsize=[8,6]) 
plt.plot(history.history['accuracy'],'r',linewidth=3.0) 
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0) 
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18) 
plt.xlabel('Epochs ',fontsize=16) 
plt.ylabel('Accuracy',fontsize=16) 
plt.title('Accuracy Curves',fontsize=16)
plt.show()