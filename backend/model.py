import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class TitanicLogisticRegression:
    def __init__(self):
        self.train = pd.read_csv('./data/titanic_train.csv')
    

    def visualise(self):
        # visualisation
        train = copy.deepcopy(self.train)

        sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

        sns.set_style('whitegrid')
        sns.countplot(x='Survived', data=train)

        sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

        sns.countplot(x='Survived', hue='Pclass', data=train)

        sns.displot(train['Age'].dropna(), kde=False, bins=30)

        train['Age'].plot.hist(bins=35)

        sns.countplot(x='SibSp', data=train)

        train['Fare'].hist(bins=40, figsize=(10,4))

        plt.figure(figsize=(10,7))
        sns.boxplot(x='Pclass', y='Age', data=train)


    def preprocess(self, train):
        # preprocessing
        train['Age'] = train[['Age', 'Pclass']].apply(self.impute_age, axis=1)

        train.drop('Cabin', axis=1, inplace=True)

        train.dropna(inplace=True)

        sex = pd.get_dummies(train['Sex'], drop_first=True)
        embark = pd.get_dummies(train['Embarked'], drop_first=True)

        train = pd.concat([train, sex, embark], axis=1)
        train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
        train.drop('PassengerId', axis=1, inplace=True)

        print(train.head())

        return train


    def train_model(self):
        # prediction
        train = self.preprocess(self.train)

        X = train.drop('Survived', axis=1)
        y = train['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        logmodel = LogisticRegression()
        logmodel.fit(X_train, y_train)

        predictions = logmodel.predict(X_test)


        # validation
        print(confusion_matrix(y_test, predictions))
        print('\n')
        print(classification_report(y_test, predictions))


        # save model
        pickle.dump(logmodel, open("model.pickle", 'wb'))


    def predict(self, input_data):
        logmodel = pickle.load(open("model.pickle", 'rb'))

        predictions = logmodel.predict(input_data)

        result_dict = {
            0: False,
            1: True,
        }

        return {"Survived": result_dict[predictions[0]]}


    def impute_age(self, cols):
        Age = cols[0]
        Pclass = cols[1]
        
        if pd.isnull(Age):
            
            if Pclass == 1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        
        else:
            return Age
