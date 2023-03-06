from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# loading data
music_dt  =pd.read_csv('music.csv')

# display the data
# print(music_dt)

# prepare 2 groups (features, output)
X=music_dt.drop(columns=['genre']) # sample features age,gender
Y=music_dt['genre'] # sample output 'genre

model = DecisionTreeClassifier()
model.fit(X,Y) # load features and sample data
predictions= model.predict([[21,1],[36,0]]) # make prediction base on the features and samp output
print(dir(model))
# print(predictions)