import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split  # to obtain train and test datasets for the same file
from sklearn.utils import shuffle

np.set_printoptions(linewidth=3000)
pd.set_option('display.max_colwidth', 3000)

print(tf.version.VERSION)

trainingDataset = pd.read_csv("dataset/train.csv")
print(trainingDataset.columns)
print(trainingDataset.head(10))




# trainingDataset = shuffle(trainingDataset)



# # Get The Features Columns
# attributes = trainingDataset.columns.tolist()
# attributes.remove('Survived')
#
# trainingX = trainingDataset[attributes]
# trainingY = trainingDataset['Survived']
#
# # Split Training and Testing Dataset
#
# Xtrain, Ytrain, Xtest, Ytest = train_test_split(trainingX, trainingY, test_size=0.33, random_state=42)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=(len(attributes)-1), activation="sigmoid")
# ])
# model.compile(optimizer="adam", loss=tf.keras.losses.binary_crossentropy)
# model.fit(Xtrain, Ytrain, epochs=10)
#
# model.evaluate(Xtest, Ytest)
