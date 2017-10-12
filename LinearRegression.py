import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def clean_data(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna('S')
    
    data.loc[data['Sex'] == 'male','Sex'] = 0
    data.loc[data['Sex'] == 'female','Sex'] = 1
    
    data.loc[data['Embarked'] == 'S','Embarked'] = 0
    data.loc[data['Embarked'] == 'C','Embarked'] = 1
    data.loc[data['Embarked'] == 'Q','Embarked'] = 2 
    
    return data



def createXMatrix(columnNames,data):
    return data.as_matrix(columnNames)

def createYMatrix(columnName,data): 
    target = data[columnName].values
    return target.reshape(target.shape[0],1)
    


if __name__ == '__main__':
   
   data_train = pd.read_csv('~/Workspace/Titanic/data/train.csv')
   data_test = pd.read_csv('~/Workspace/Titanic/data/test.csv')
     
   predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

   data = clean_data(data_train)
   xMatrix = createXMatrix(predictors,data)
   yMatrix = createYMatrix('Survived',data)
   print yMatrix.shape



