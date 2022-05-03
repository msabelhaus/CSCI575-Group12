import numpy as np
from readData import ParseData, relabel, readImages

def loadData(dataFile, IMSIZE, trainAmount, readFiles=False):
    if(readFiles):
        trainImages, trainLabels, testImages, testLabels = readImageDataFiles()
    else:
        allData = ParseData(dataFile)
        allData = allData
        allKeys = [x[0] for x in allData]
        allLabels = [x[1] for x in allData]
        #lables need to be relabled because the 'to_catagorical' function needs the labels
        # to be labled numerically in order (i think)
        allLabels = relabel(allLabels)

        trainKeys = np.array(allKeys[0:trainAmount])
        trainImages = readImages(trainKeys, IMSIZE)

        trainLabels = np.array(allLabels[0:trainAmount])
        
        testKeys = np.array(allKeys[trainAmount-1:-1])
        testImages = readImages(testKeys, IMSIZE )
        testLabels = np.array(allLabels[trainAmount-1:-1])

        #######################################
        # Write the data to a file for use to avoid rereading the data
        #######################################
        writeImageDataFiles(trainImages, trainLabels, testImages, testLabels)

    classCount = len(np.unique(trainLabels))
    return trainImages, trainLabels, testImages, testLabels, classCount

def writeImageDataFiles(trainImages, trainLabels, testImages, testLabels):
    #######################################
    # Write the data to a file for use to avoid rereading the data
    #######################################
    # Save train images
    file = open('.\\readFiles\\trainImages', 'wb')
    np.save(file, trainImages)
    file.close
    # Save train labels
    file = open('.\\readFiles\\trainLabels', 'wb')
    np.save(file, trainLabels)
    file.close
    # Save test images
    file = open('.\\readFiles\\testImages', 'wb')
    np.save(file, testImages)
    file.close
    # Save test labels
    file = open('.\\readFiles\\testLabels', 'wb')
    np.save(file, testLabels)
    file.close

def readImageDataFiles():
    #######################################
    # Read the data from a file for use to avoid rereading the data
    #######################################
    # Save train images
    file = open('.\\readFiles\\trainImages', 'rb')
    trainImages = np.load(file)
    file.close
    # Save train labels
    file = open('.\\readFiles\\trainLabels', 'rb')
    trainLabels = np.load(file)
    file.close
    # Save test images
    file = open('.\\readFiles\\testImages', 'rb')
    testImages = np.load(file)
    file.close
    # Save test labels
    file = open('.\\readFiles\\testLabels', 'rb')
    testLabels = np.load(file)
    file.close

    return trainImages, trainLabels, testImages, testLabels
    