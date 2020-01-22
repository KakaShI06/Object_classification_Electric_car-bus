from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier=Sequential()   # Intialializing CNN

# Convolution Step
classifier.add(Conv2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu')) # order of  input_shape for tensorflow backend

#Pooling Step
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size= (2,2)))

#Flattening Step
classifier.add(Flatten()) 

#Full Connection Step
classifier.add(Dense(units= 128, activation= 'relu'))
classifier.add(Dense(units= 1, activation= 'sigmoid')) #output layer 


# Compiling the CNN

classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

path= "Training_set/Training_set"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


training_set = train_datagen.flow_from_directory( path,
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

path="test/test"

test_set = test_datagen.flow_from_directory(
                                              path,
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')
                                              
classifier.fit_generator(
        training_set,
        steps_per_epoch=1408,
        epochs=5,
        validation_data=test_set,
        validation_steps=1480)