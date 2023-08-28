from flask import Flask, render_template, request
import model_loader
import numpy as np
import pandas as pd

app = Flask(__name__)

models = {
    'model1': model_loader.load_model('model1.pkl'),
    'model2': model_loader.load_model('model2.pkl'),
    'model3': model_loader.load_model('model3.pkl')
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model/<model_name>', methods=['GET', 'POST'])
def predict(model_name):
    if request.method == 'POST':
        input_data = []
        if model_name == 'model1':
            internships = float(request.form['Internships'])
            cgpa = float(request.form['CGPA'])
            input_data = np.asarray([[internships, cgpa]]).reshape(1,2)
            print(input_data)
            
        elif model_name == 'model2':
            input_data = np.asarray([float(request.form['analytical']), float(request.form['english']),
                          float(request.form['aptitude']), float(request.form['coding'])]).reshape(1,4)
        elif model_name == 'model3':
            input_data = np.asarray([float(request.form['ssc_p']), float(request.form['degree_p']),
                          float(request.form['hsc_p']), float(request.form['workex'])]).reshape(1,4)
        
        model = models[model_name]
        prediction = model.predict([input_data][0])[0]
        result = "You will be not be placed" if prediction == 0 else "You will be placed "
        return render_template('model.html', model_name=model_name, prediction=result)
    
    return render_template('model.html', model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True)
