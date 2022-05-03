import matplotlib.pyplot as plt

def plotImage(image, figName):
    plt.figure(figsize=[10,5])

    #Display the first image in training data
    plt.subplot(121)
    singleTrain = image
    plt.imshow(singleTrain, cmap='gray')
    plt.imshow(image[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(figName))
    plt.savefig(str(figName) + '.png')


#Plot the Loss Curves
def plotLossCurves(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

#Plot the Accuracy Curves
def plotAccuracyCurves(history):
    plt.figure(figsize=[8,6]) 
    plt.plot(history.history['accuracy'],'r',linewidth=3.0) 
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0) 
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18) 
    plt.xlabel('Epochs ',fontsize=16) 
    plt.ylabel('Accuracy',fontsize=16) 
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()  