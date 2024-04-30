from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

def cleaning(data):
   data['Feedback'] = data['Feedback'].apply(lambda x: 'Positive' if 'P' in str(x) or 'p' in str(x) else x)
   data['Feedback'] = data['Feedback'].apply(lambda x: 'Negative' if 'N' in str(x) or 'n' in str(x) else x)
   data['Feedback'] = data['Feedback'].apply(lambda x:  np.nan if x != 'Positive' and x != 'Negative' else x)


   data.loc[data['Age'] > 50, 'Age'] = np.nan

   return data



try:
    with open('Voting_model80.pkl', 'rb') as f:
        food_pipeline = joblib.load(f)
except Exception as err:
    print(f"Unexpected error: {err}, Type: {type(err)}")


app = Flask(__name__)

@app.route('/')
def home():
    return "Run Seccessfully...."
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
        new_data_cleaned = food_pipeline.named_steps['cleaning'].transform(data)

        new_data_transformed = food_pipeline.named_steps['preprocessor'].transform(new_data_cleaned)

        predictions = food_pipeline.named_steps['classifier'].predict(new_data_transformed)

        return jsonify({'Prediction: ': predictions.tolist()})
    except Exception as err:
        return jsonify({"Unexpected error: ": str(err)})



if __name__ == '__main__':
    app.run(debug = True)
