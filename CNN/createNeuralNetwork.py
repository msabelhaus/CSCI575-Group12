from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, losses
from tensorflow.keras.layers import Dropout, Dense

def createNeuralNetwork(IMSIZE, channels, optimizer, classCount):
    model = Sequential()
    model.add(layers.Conv2D(32,(3, 3), activation='relu', input_shape=(IMSIZE[0], IMSIZE[1], channels)))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))


    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    model.add(layers.Dense(classCount))
    model.add(Dropout(0.3))
    model.compile(optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    print('Summary for CNN using ',optimizer )
    
    model.summary()


    return model