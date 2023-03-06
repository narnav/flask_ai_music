from flask import Flask, render_template, request
from flask_cors import CORS
from sklearn.tree import DecisionTreeClassifier
import joblib
import csv
import pandas as pd

app = Flask(__name__)
CORS(app)
@app.route("/")
def hello_world():
    return render_template('main.html')

@app.route("/predictions",methods=['GET','POST'])
def predicti():
    predictions=[]
    if request.method == 'POST':
        gender=request.form.get('gender')
        age=request.form.get('age')

        model=joblib.load( 'our_pridction.joblib')
        predictions= model.predict([[age,gender]])
        # print(predictions)
        # print(gender)
    return render_template('predictions.html',msg=predictions)

@app.route('/add', methods=['GET', 'POST'])
def add_fav_genere():
    if request.method == 'POST':
        age=request.form.get('age')
        gender=request.form.get('gender')
        genre=request.form.get('genre')
        # print(f'age{age},gender{gender},genere{genre}')
        f = open('music.csv', 'a',newline='')
        writer = csv.writer(f)
        data = [age, gender, genre]
        writer.writerow(data)
        f.close()
        return render_template("add.html", msg=f'age{age},gender{gender},genere{genre}' )
    return render_template("add.html")

@app.route("/learn",methods=['GET'])
def learn():
    music_dt  =pd.read_csv('music.csv')
    X=music_dt.drop(columns=['genre']) # sample features age,gender
    Y=music_dt['genre'] # sample output 'genre
    model = DecisionTreeClassifier()
    model.fit(X,Y) # load features and sample data
    joblib.dump(model, 'our_pridction.joblib') #binary file
    return render_template('learn.html',msg="learned")

if __name__ == '__main__':
    app.run(debug=True)