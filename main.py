import tensorflow as tf
import numpy as np
import pandas as pd

np.set_printoptions(linewidth=3000)
pd.set_option('display.max_colwidth', 3000)
print(tf.version.VERSION)


dataset = pd.read_csv("dataset/train.csv")

print(dataset.columns)
print(dataset.head(5))
print(dataset.shape)

