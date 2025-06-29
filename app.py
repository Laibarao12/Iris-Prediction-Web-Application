from flask import Flask
from flask import render_template, request
import joblib

app= Flask(__name__)


model = joblib.load('iris_model.pkl')



@app.route("/",methods=['GET'])  
def home():
   return render_template('page.html')
  

@app.route('/predict', methods=['POST']) 
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)

    species_dict = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    species_image_map = {
        'setosa': 'setosa.png',
        'versicolor': 'versicolor.png',
        'virginica': 'virginicia.png'
    }

    predicted_species = species_dict.get(prediction[0], "Unknown")
    image_filename = species_image_map.get(predicted_species.lower(), 'default.png')

    return render_template('result.html', species=predicted_species, image_file=image_filename)


if __name__=="__main__":
    app.run(debug=True)