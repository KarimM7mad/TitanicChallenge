import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.set_printoptions(linewidth=3000)
pd.set_option('display.max_colwidth', 3000)

print(tf.version.VERSION)


def checkTitle(x):
    if ("Mr." in x["Name"]) or ("Master." in x["Name"]) or ("Don." in x["Name"]) or ("Major." in x["Name"]) \
            or ("Rev." in x["Name"]) or ("Capt." in x["Name"]) or ("Jonkheer." in x["Name"]) or ("Sir." in x["Name"]):
        x = 1
    elif ("Ms." in x["Name"]) or ('Mlle.' in x["Name"]) or ('Miss.' in x["Name"]) or ('Lady.' in x["Name"]):
        x = 2
    elif ("Mme." in x["Name"]) or ("Countess." in x["Name"]) or ('Mrs.' in x["Name"]):
        x = 3
    elif ("Dr." in x["Name"]) or ("Col." in x["Name"]):
        if x["Sex"] == 1:
            x = 1
        else:
            x = 2
    else:
        x = 5
    return x


def prepareDataSet():
    trainingDataset = pd.read_csv("dataset/train.csv")
    # 0 : female, 1 : male
    trainingDataset.at[trainingDataset["Sex"] == "male", "Sex"] = 1
    trainingDataset.at[trainingDataset["Sex"] == "female", "Sex"] = 0
    # 1 : S, 2 : C, 3 : Q
    trainingDataset.at[trainingDataset["Embarked"] == 'S', 'Embarked'] = 1
    trainingDataset.at[trainingDataset["Embarked"] == 'C', 'Embarked'] = 2
    trainingDataset.at[trainingDataset["Embarked"] == 'Q', 'Embarked'] = 3
    # 1 : mr, 2:mrs, 3:miss
    trainingDataset["Name"] = trainingDataset.apply(checkTitle, axis=1)
    trainingDataset.drop(columns=["Cabin", "Ticket","PassengerId"], inplace=True)

    trainingDataset = trainingDataset.drop(trainingDataset[trainingDataset.Fare == 0].index)

    trainingDataset.dropna(axis=0, inplace=True)

    trainingDataset["covAgeANDPclass"] = trainingDataset["Age"] * trainingDataset["Pclass"]
    trainingDataset["familySize"] = trainingDataset["SibSp"] + trainingDataset["Parch"]

    trainingDataset.at[trainingDataset["familySize"] == 0, "familySize"] = 1
    trainingDataset["pricePerPerson"] = trainingDataset["Fare"] / trainingDataset["familySize"]

    print(trainingDataset.columns)

    return trainingDataset


def startML(DataSet):
    # Get The Features Columns
    attributes = DataSet.columns.tolist()
    attributes.remove('Survived')
    trainingX = DataSet[attributes]
    trainingY = DataSet['Survived']

    scalar = StandardScaler()
    scalar.fit(trainingX)
    trainingX = scalar.transform(trainingX)

    # Split Training and Testing Dataset for Machine Learning Part
    trainingFeatures, devsetFeatures, trainingLabels, devsetLabels = train_test_split(trainingX, trainingY, test_size=0.08, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=24, input_shape=(len(attributes),), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(units=12, activation=tf.nn.tanh),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ])

    optimizer = tf.keras.optimizers.Adam(lr=0.006)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

    model.fit(trainingFeatures, trainingLabels, validation_split=0.25, epochs=10)

    print("--------------------------")
    model.evaluate(devsetFeatures, devsetLabels)
    # model.summary()
    return model


# Execution Part
Dataset = prepareDataSet()
startML(Dataset)
