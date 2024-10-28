from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

app = Flask(__name__)

# Load the machine learning model
loaded_model = joblib.load('mlp_classifier_model.joblib')

# Initialize a StandardScaler
scaler = StandardScaler()

# Initialize a LabelEncoder for 'time_of_day' and 'food_type'
time_of_day_encoder = LabelEncoder()
food_type_encoder = LabelEncoder()

# Define the label encodings (assuming you used 0 for Morning, 1 for Afternoon, and 2 for Evening)
time_of_day_encoder.classes_ = np.array(['Morning', 'Afternoon', 'Evening'])

# Define the label encodings (assuming you used 0 for Vegetable, 1 for Meat, and 2 for Fruit)
food_type_encoder.classes_ = np.array(['Vegetable', 'Meat', 'Fruit'])

def preprocess_input_data(temperature, humidity, gas_levels, time_of_day, food_type, storage_duration):
    # Encode categorical variables
    time_of_day_encoded = time_of_day_encoder.transform([time_of_day])[0]
    food_type_encoded = food_type_encoder.transform([food_type])[0]
    
    # Combine all features into an array
    input_data = np.array([temperature, humidity, gas_levels, time_of_day_encoded, food_type_encoded, storage_duration])
    
    # Standardize the feature values
    input_data = scaler.transform([input_data])
    
    return input_data

@app.route('/')
def show_form():
    return render_template('food_spoilage_form.html')

@app.route('/classify', methods=['POST'])
def classify():
    print("Received POST request to /classify")
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    gas_levels = float(request.form['gas_levels'])
    time_of_day = request.form['time_of_day']
    food_type = request.form['food_type']
    storage_duration = int(request.form['storage_duration'])

    # Fit the StandardScaler with the current input data
    input_data = np.array([temperature, humidity, gas_levels, 0, 0, storage_duration]).reshape(1, -1)  # Note: Using dummy values for time_of_day and food_type
    scaler.partial_fit(input_data)

    # Preprocess the input data
    input_data = preprocess_input_data(temperature, humidity, gas_levels, time_of_day, food_type, storage_duration)

    # Predict spoilage
    spoilage_result = loaded_model.predict(input_data)
    
    # Print the prediction to the console
    print(f"The food is classified as: {'Spoiled' if spoilage_result[0] == 1 else 'Not Spoiled'}")
    
    # This line sends the result to the template, but it won't affect what's printed in the terminal.
    return render_template('result.html', spoilage_result=spoilage_result[0])


if __name__ == '__main__':
    app.run(debug=True)
