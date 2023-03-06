from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# gather data
music_dt  =pd.read_csv('music.csv')


# display the data
# music_dt


# prepare 2 groups
X=music_dt.drop(columns=['genre']) # sample features
Y=music_dt['genre'] # sample output


# X= input train,test, Y = output train, testing
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=.2)


model = DecisionTreeClassifier()
model.fit(X_train,Y_train) # load features and sample data
joblib.dump(model, 'our_pridction.joblib') #binary file
predictions= model.predict(X_test) # make prediction base on the features and samp output
score=accuracy_score(Y_test,predictions)
print( score)
