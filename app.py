from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

root_directory = app.root_path
print(root_directory)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    sepal_length = float(request.form.get('sepal length (cm)'))
    sepal_width = float(request.form.get('sepal width (cm)'))
    petal_length = float(request.form.get('petal length (cm)'))
    petal_width = float(request.form.get('petal width (cm)'))

    # prediction
    result = model.predict(
        np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, 4))

    if result[0] == 0:
        result = 'SETOSA'
    elif result[0] == 1:
        result = 'VERSICOLOR'
    else:
        result = "VERGINICA"

    return render_template('index.html', result=result)


if (__name__) == '__main__':
    app.run(host="0.0.0.0", port=3000)
