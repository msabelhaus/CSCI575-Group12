import csv
import numpy as np
import cv2

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
    
    labeled_list = [int(numeric_string) for numeric_string in labeled_list]
    return labeled_list

#read images from list of keys 
def readImages(imageKey, IMSIZE):
    folder = '..\\data\\google-data\\images\\'
    images = np.ndarray([len(imageKey), IMSIZE[0],IMSIZE[1],3])
    imIdx = 0
    for key in imageKey:
        filename = folder + key + '.jpg'
        rawim = cv2.imread(filename)
        imresize = cv2.resize(rawim,IMSIZE)
        images[imIdx,:,:,:] = imresize
        imIdx = imIdx+1
    return images

