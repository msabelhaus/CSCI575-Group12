import sys, os, multiprocessing, csv
from cv2 import BFMatcher_create

from grpc import channel_ready_future
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image
import cv2
import time

# import custom made modules
from loadData import loadData
from plotData import plotImage, plotLossCurves, plotAccuracyCurves
from createNeuralNetwork import createNeuralNetwork

IMSIZE = (32, 32)
TRAINAMOUNT = 8000
dataFile = '..\\data\\google-data\\df_final.csv'
modelType = 'CNN'
optimizer = 'rmsprop'
batchSize = 50
channels = 3
epochs = 40
verbose = 1
readFiles = False

trainImages, trainLabels, testImages, testLabels, classCount = loadData(dataFile, IMSIZE, TRAINAMOUNT, readFiles)
model = createNeuralNetwork(IMSIZE, channels, optimizer, classCount) 
start = time.time()
history = model.fit(trainImages, trainLabels,  epochs=epochs, verbose=verbose,validation_data=(testImages, testLabels))
end = time.time()
print("Time consumed in working: ",end - start)
[test_loss, test_acc] = model.evaluate(testImages, testLabels)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print("Time elapsed on working...")

plotLossCurves(history)
plotAccuracyCurves(history)