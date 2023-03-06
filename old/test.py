from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

model=joblib.load( 'our_pridction.joblib')
predictions= model.predict([[21,1]])
print(predictions)
