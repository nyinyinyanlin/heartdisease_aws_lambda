import boto3
import os
import ctypes
import uuid
import sklearn
import pickle
import numpy as np

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a'):
            continue
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

s3_client = boto3.client('s3')

def handler(event, context):

    #Info
    age = float(event.get('record')['age'])
    sex = float(event.get('record')['sex'])
    cp = float(event.get('record')['cp'])
    trestbps = float(event.get('record')['trestbps'])
    chol = float(event.get('record')['chol'])
    fbs = float(event.get('record')['fbs'])
    restecg = float(event.get('record')['restecg'])
    thalach = float(event.get('record')['thalach'])
    exang = float(event.get('record')['exang'])
    oldpeak = float(event.get('record')['oldpeak'])
    slope = float(event.get('record')['slope'])
    ca = float(event.get('record')['ca'])
    thal = float(event.get('record')['thal'])
    
    #Row
    record = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,13)

    #load model
    bucket = 'heartdisease_bucket'
    key = 'model_tuple.pkl'
    download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    s3_client.download_file(bucket, key, download_path)
    
    f = open(download_path, 'rb')
    (scaler, model) = pickle.load(f)
    record = scaler.transform([record])
    f.close()
    
    #Predict
    class_predicted = model.predict(record)[0]
    return class_predicted
