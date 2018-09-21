# Heart Disease Prediction System - AWS Lambda Function

This lambda function is supposed to setup alongside with AWS S3 for storing trained model and Gateway API to expose a public REST API path. The function takes in 13 attributes and return a single integer value of either 0 or 1 as an output based on the possibility of having a heart disease.

### Trained Model

The model object is pickled as model_tuple.pkl from a Python tuple of `(scaler, svc_model)` because the model is trained using a standardized/scaled dataset and thus, data to be predicted also needed to be standardized.

### AWS Note
Bucket should be named `heartdisease_bucket` and lambda function should be named `heartdisease`. `model_tuple.pkl` is the pickled trained model with scaler included and should be uploaded to `heartdisease_bucket` bucket with the same name.
