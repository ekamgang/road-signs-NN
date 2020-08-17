import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

        # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

        # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
            (images), (labels), test_size=TEST_SIZE
        )
        # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    f = os.listdir(data_dir)    #list all subcategories
    images=[]
    label = []
    dim=(IMG_WIDTH, IMG_HEIGHT)
    for files in f:     #iterate through eachcategory
        name= (files)
        fileName = "gtsrb/"+files
        img= os.listdir(fileName)
        for image in img:
                #im = open(image, "r")
                imageName= 'gtsrb\\'+files+'\\'+image
                i = cv2.imread(imageName,1)
                #cv2.imshow("image", i)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                im=cv2.resize(i, dim, interpolation = cv2.INTER_AREA)
            #images.apnpend(np.array(Image.open(image)).reshape(IMG_HEIGHT,IMG_WIDTH,3) for image in open(files,"r"))
                images.append(im)
                label.append((int(name)))
    return ((images),(label))


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(

            # Convolutional layer. Learn 32 filters using a 3x3 kernel
            #32, (3, 3), 
            activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add an output layer with output units for all 10 digits
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

        # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError



if __name__ == "__main__":
    main()
