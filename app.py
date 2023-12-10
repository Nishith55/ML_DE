from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define the features used for prediction
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "PPA Tenure(Y)", "IRR"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_features = [float(request.form[feature]) for feature in features]

    # Create a DataFrame from the input features
    input_df = pd.DataFrame([input_features], columns=features)

    # Make predictions
    prediction = model.predict(input_df)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
