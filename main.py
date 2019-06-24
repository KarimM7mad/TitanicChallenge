import numpy as np
import pandas as pd
import tensorflow as tf
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


def prepareDataSet(dataSet):
    # 0 : female, 1 : male
    dataSet.at[dataSet["Sex"] == "male", "Sex"] = 1
    dataSet.at[dataSet["Sex"] == "female", "Sex"] = 0
    # 1 : S, 2 : C, 3 : Q
    dataSet.at[dataSet["Embarked"] == 'S', 'Embarked'] = 1
    dataSet.at[dataSet["Embarked"] == 'C', 'Embarked'] = 2
    dataSet.at[dataSet["Embarked"] == 'Q', 'Embarked'] = 3
    # 1 : mr, 2:mrs, 3:miss
    dataSet["Name"] = dataSet.apply(checkTitle, axis=1)
    dataSet.drop(columns=["Cabin", "Ticket", "PassengerId"], inplace=True)

    dataSet = dataSet.drop(dataSet[dataSet.Fare == 0].index)

    dataSet.dropna(axis=0, inplace=True)

    dataSet["covAgeANDPclass"] = dataSet["Age"] * dataSet["Pclass"]
    dataSet["familySize"] = dataSet["SibSp"] + dataSet["Parch"]

    dataSet.at[dataSet["familySize"] == 0, "familySize"] = 1
    dataSet["pricePerPerson"] = dataSet["Fare"] / dataSet["familySize"]

    # Get The Features Columns
    attributes = dataSet.columns.tolist()
    attributes.remove('Survived')
    X = dataSet[attributes]
    y = dataSet['Survived']
    # Data Normalization
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    return (X, y)


def CreateModel(trainingX, trainingY):
    # Split Training and Testing Dataset for Machine Learning Part
    # trainingFeatures, devsetFeatures, trainingLabels, devsetLabels = train_test_split(trainingX, trainingY, test_size=0.08, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=24, input_shape=(trainingX.shape[1],), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(units=12, activation=tf.nn.tanh),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ])

    optimizer = tf.keras.optimizers.Adam(lr=0.006)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

    model.fit(trainingX, trainingY, validation_split=0.33, epochs=10)

    print("--------------------------")
    return model


# Execution Part
# Training with CV
trainingDataset = prepareDataSet(pd.read_csv("dataset/train.csv"))
model = CreateModel(trainingDataset[0], trainingDataset[1])
# Testing
testDataset = pd.concat([pd.read_csv("dataset/test.csv"), pd.read_csv("dataset/gender_submission.csv")['Survived']], axis=1)
print(testDataset.shape)

testDataset = prepareDataSet(testDataset)

print("-------------=============---------------")
model.evaluate(testDataset[0], testDataset[1])
