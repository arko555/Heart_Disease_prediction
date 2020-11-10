from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
import os

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's check if someone has heart disease 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
        description: Age of User 
      - name: sex
        in: query
        type: number
        required: true
        description: Sex of user 1:Male 0:Female
        enum: [1,0]
      - name: chest_pain
        in: query
        type: number
        required: true
        description: 1 = typical angina, Value 2 = atypical angina, Value 3 = non-anginal pain, Value 4 = asymptomatic
        enum: [1,2,3,4]
      - name: resting_blood_pressure
        in: query
        type: number
        required: true
        description: Person's resting blood pressure (mm Hg on admission to the hospital)
      - name: cholestrol
        in: query
        type: number
        required: true
        description: The person's cholesterol measurement in mg/dl
      - name: fasting_blood_pressure
        in: query
        type: number
        required: true
        description: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
        enum: [0,1]
      - name: rest_eloctocaridiogram_measure
        in: query
        type: number
        required: true
        description: Resting electrocardiographic measurement(0 = normal, 1 = having ST-T wave abnormality, 2 = probable or             definite left ventricular hypertrophy
        enum: [0,1,2]
      - name: max_heart_rate_achieved
        in: query
        type: number
        required: true
        description: The person's maximum heart rate achieved (usually between 71-202)
      - name: exercise_induced_angina
        in: query
        type: number
        required: true
        description: Exercise induced angina (1 = yes; 0 = no)
        enum: [0,1]
      - name: oldpeak
        in: query
        type: number
        required: true
        description: ST depression induced by exercise relative to rest usually between 0-2
      - name: slope
        in: query
        type: number
        required: true
        description: the slope of the peak exercise ST segment (Value 1 = upsloping, Value 2 = flat, Value 3 = downsloping)
        enum: [1,2,3]
      - name: number_major_vessels
        in: query
        type: number
        required: true
        description: The number of major vessels (0,1,2,3,4)
        enum: [0,1,2,3,4]
      - name: thalasemia
        in: query
        type: number
        required: true
        description: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
        enum: [0,1,2,3]
              
    responses:
        200:
            description: The output values
        
    """
    age=request.args.get("age")
    sex=request.args.get("sex")
    chest_pain=request.args.get("chest_pain")
    resting_blood_pressure=request.args.get("resting_blood_pressure")
    cholestrol=request.args.get("cholestrol")
    fasting_blood_pressure=request.args.get("fasting_blood_pressure")
    rest_eloctocaridiogram_measure=request.args.get("rest_eloctocaridiogram_measure")
    max_heart_rate_achieved=request.args.get("max_heart_rate_achieved")
    exercise_induced_angina=request.args.get("exercise_induced_angina")
    oldpeak=request.args.get("oldpeak")
    slope=request.args.get("slope")
    number_major_vessels=request.args.get("number_major_vessels")
    thalasemia=request.args.get("thalasemia")
    
    prediction=classifier.predict([[age,sex,chest_pain,resting_blood_pressure,cholestrol,fasting_blood_pressure,
                                    rest_eloctocaridiogram_measure,max_heart_rate_achieved,exercise_induced_angina,
                                    oldpeak,slope,number_major_vessels,thalasemia]])
    
    if prediction==1:
        return "The person has Heart Disease"
    else:
        return "The person doesn't have Heart Disease"

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's check if someone has heart disease 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    #host = os.getenv('host')
    #port = os.getenv('port')
    app.run(host='0.0.0.0', port=8000)
    
    
    
    
    
    
    
    
    
    
    
