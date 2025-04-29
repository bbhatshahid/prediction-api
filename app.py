from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load models
rf_model = joblib.load('RFmodel.pkl')
lr_model = joblib.load('LRModel.pkl')
xgb_model = joblib.load('XGBmodel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        features = [
            float(data['bedrooms']),
            float(data['bathrooms']),
            float(data['livingarea']),
            float(data['floors']),
            float(data['arhouse']),
            float(data['builtyr']),
            float(data['lotarea']),
            float(data['grade']),
            float(data['waterfront']),
        ]

        input_data = np.array(features).reshape(1, -1)

        rf_prediction = float(rf_model.predict(input_data)[0])
        lr_prediction = float(lr_model.predict(input_data)[0])
        xgb_prediction = float(xgb_model.predict(input_data)[0])

        return jsonify({
            'RandomForest': round(rf_prediction, 2),
            'LinearRegression': round(lr_prediction, 2),
            'XGBoost': round(xgb_prediction, 2)
        })
    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return 'Server is running!'

if __name__ == '__main__':
    app.run(port=5001, debug=True)
