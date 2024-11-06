#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('C:/Users/HP/Downloads/stock-prediction-app/stock_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Extract features from the data
        features = np.array(data['features']).reshape(1, -1)
        
        # Predict stock price
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[ ]:




