import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Filenames
MODEL_FILENAME = 'logistic_regression_model.joblib'
SCALER_FILENAME = 'standard_scaler.joblib'

# -------------------------------
# Save model and scaler (run once after training)
# -------------------------------
def save_model(model, scaler):
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")
    print(f"Scaler saved to {SCALER_FILENAME}")


# -------------------------------
# Load model and scaler
# -------------------------------
def load_model():
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    return model, scaler


# Load once when server starts (better performance)
model, scaler = load_model()


# -------------------------------
# API endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key'}), 400

        features = data['features']  # Example: [1.2, 3.4, 5.6]

        # Scale input
        scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled)

        return jsonify({
            'prediction': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------------------
# Run app
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
