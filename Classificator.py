import tensorflow as tf
from tensorflow import keras
import os
import cv2 as cv
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

image_exts = ['jpeg', 'jpg']    # add 'png' to the list if required.

# filters out any images with undesirable image formats, currently only accepts jpg, Jpeg
def training_image_clean(data):
    for image_class in os.listdir(data):
        print(image_class)
        for image in os.listdir(os.path.join(data, image_class)):
            image_path = os.path.join(data, image_class, image)
            img = cv.imread(image_path)
            try:
                img = cv.imread(image_path)
                tip = imghdr.what(image_path) # checks for image format
                if tip not in image_exts:
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path) # removes image if either test fails
            except Exception as e:
                print('Issue with image{}'.format(image_path))
    preProcess(data)

# Image Preprocessing, seperates all data into batches of 32, assigns colour type : RGB, and resizes them to be (256x256)
def preProcess(data_dir):
    for image_class in os.listdir(data_dir):
        print(image_class)
        counter = 0
        for image in os.listdir(os.path.join(data_dir, image_class)):
            counter += 1
        print(counter)
    data = tf.keras.utils.image_dataset_from_directory(data_dir, color_mode='rgb', interpolation="nearest" , image_size = (256, 256))
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    print(len(batch))
    print(batch[1]) # checks to make sure its being batched correctly, with shuffled data in it
    # 0 = buildings
    # 1 = Food
    # 2 = landscapes
    # 3 = people 
    print(batch[0].shape)

# Scaling the data from 0-255 rgb values to 0-1
    print(f'Pre-Scaled Iterator max: {data_iterator.next()[0].max()}')
    print(f'Pre-Scaled Iterator min: {data_iterator.next()[0].min()}')
    data = data.map(lambda x,y: (x/255,y))
    scaled_iterator = data.as_numpy_iterator()
    print(f'Scaled Iterator max: {scaled_iterator.next()[0].max()}')
    print(f'Scaled Iterator min: {scaled_iterator.next()[0].min()}')

# Segmenting data into Training - Validating - Testing data
    print(f'DATA SIZE: {len(data)}')
    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2) + 1
    test_size = int(len(data)*.1)
    print(f'TRAINING SIZE: {train_size}')
    print(f'VALIDATION SIZE: {val_size}')
    print(f'TEST SIZE: {test_size}')
    print(f'TOTAL SIZE: {train_size+val_size+test_size}')
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    buildModel(train, val, test)

# Building the model, 2 Conv2D layers, 2 pooling Layers, 1 flatten Layer, 2 Dense Layers
def buildModel(train, val, test):

    model = Sequential()

    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1,1), activation = "relu", input_shape = (256,256,3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) # down samples the input into smaller components by finding average value across the area

    model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D())    

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation = "relu"))
    model.add(tf.keras.layers.Dense(4, activation = "softmax"))

# Compiling model with adam optimizer, learning rate 0.001, sparse Categorical crossentropy
    model.compile('adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    print(model.summary()) # checks model architecture
    callbacks = keras.callbacks.ModelCheckpoint("LargeSavedModel") # assignes what it will be saves under
    model.fit(train, epochs = 25, callbacks=callbacks, validation_data = val)
# ^^^^^^^^^^^^^^^ Trains the model (Training data, epochs it will be run for, What itll be saved as, Validation data)



# Runs the entire process from start till finish
def runSystem(data):
    training_image_clean(data)


# for experiment, interchange the training imageData with: 

data_dir_small = 'SmallData' 
data_dir_large = 'LargeData'
data_set = ""

# change parameter with the dataset you want to build with
#runSystem(data_dir_large)



# PERSONAL TESTING 

def eval_image_clean(data):
    for image in os.listdir(data):
        image_path = os.path.join(data, image)
        img = cv.imread(image_path)
        try:
            img = cv.imread(image_path)
            tip = imghdr.what(image_path) 
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path) # removes image if either test fails
        except Exception as e:
            print('Issue with image{}'.format(image_path))

def evalModel():
    eval_path = "Evaluation"
    savedModel = "SmallSavedModel"
    model = tf.keras.models.load_model(savedModel)

    eval_image_clean(eval_path)

    test_data = tf.keras.utils.image_dataset_from_directory(eval_path, labels= None, batch_size=4, color_mode='rgb', interpolation="nearest", image_size=(256, 256))

    titles = ["Building", "Food", "Landscapes", "People"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    for i, (image) in enumerate(test_data):
        prediction = model.predict(image)
        decision = np.argmax(prediction, axis=1)
        categories = [titles[cat] for cat in decision]
        for j in range(len(image)):
            axes[j].imshow(image[j].numpy().astype("uint8"))
            axes[j].set_title(categories[j])
            axes[j].axis("off")

    plt.show()

evalModel() # Uncomment to test CNN with "evaluation" dataset (4 images)
             # These images can be replaces with any JPG file desired, if 
             # the images you want to replace them with are png, go to the 
             # commented line 11, and add 'png' to that list, it should still
             #  work, but for the sake of the experiment, i just used JPG and Jpeg