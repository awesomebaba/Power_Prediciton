import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('household_power_consumption.csv') 
scaler = MinMaxScaler() 

model = load_model('model_sub_meter.h5') 

def prepare_input_data(data, scaler, sequence_length):
    # Use the last sequence_length minutes of data
    data = data[-sequence_length:]
    data_scaled = scaler.transform(data)  
    return data_scaled

def predict_for_minute(data, model, scaler, sequence_length):
    input_data = prepare_input_data(data, scaler, sequence_length)
    prediction = model.predict(np.expand_dims(input_data, axis=0))
    prediction_rescaled = scaler.inverse_transform(prediction)
    return prediction_rescaled.flatten()

sequence_length = 60  
last_minute_data = df.iloc[-1].values.reshape(1, -1)  
predicted_values = predict_for_minute(last_minute_data, model, scaler, sequence_length)

print('Predicted values for the next 5 minutes:')
print(predicted_values)
