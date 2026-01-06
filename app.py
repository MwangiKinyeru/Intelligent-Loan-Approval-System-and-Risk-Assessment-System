from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import numpy as np
import os
import requests

from utils.validation import validate_input
from utils.preprocessing import preprocess_for_classification, preprocess_for_regression
from utils.kesonic import calculate_kesonic_rate

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models once
classifier = joblib.load(os.path.join(MODEL_DIR, "final_classifier_xgboost.pkl"))
clf_scaler = joblib.load(os.path.join(MODEL_DIR, "classification_scaler.pkl"))

regressor = joblib.load(os.path.join(MODEL_DIR, "best_loan_amount_predictor.pkl"))
reg_scaler = joblib.load(os.path.join(MODEL_DIR, "regression_scaler.pkl"))

kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
kesonic_scaler = joblib.load(os.path.join(MODEL_DIR, "kesonic_scaler.pkl"))

with open(os.path.join(MODEL_DIR, "cluster_info.pkl"), "rb") as f:
    cluster_info = pickle.load(f)

APPROVAL_THRESHOLD = 0.35
GOOGLE_SHEETS_URL = "https://script.google.com/macros/s/AKfycbz9A2Hml7rHgPURLmt0FQ8ICcm2qrn0V8wSfENPuWCoZ6bghb1UB8PGz9_uTC0Stlfy3w/exec"

# Serve HTML pages
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/loan", methods=["GET"])
def loan_page():
    return render_template("loan.html")

# API endpoint: Predict
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Input validation
    error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    # Classification
    X_clf = preprocess_for_classification(data, clf_scaler)

    missing_cols = set(classifier.get_booster().feature_names) - set(X_clf.columns)
    for col in missing_cols:
        X_clf[col] = 0
    X_clf = X_clf[classifier.get_booster().feature_names]

    prob = classifier.predict_proba(X_clf)[:, 1][0]
    approved = prob >= APPROVAL_THRESHOLD

    if not approved:
        response = {
            "approved": False,
            "approval_probability": round(float(prob), 4)
        }
        # Sends 0 to Google Sheets for rejected applications
        _send_to_sheets(data, response, is_final=True)  
        return jsonify(response)

    # Regression
    X_reg = preprocess_for_regression(data, reg_scaler)
    X_reg = X_reg.reindex(columns=regressor.feature_names_in_, fill_value=0)

    loan_amount_log = regressor.predict(X_reg)[0]
    loan_amount = float(np.expm1(loan_amount_log))

    # KESONIA
    kesonic_result = calculate_kesonic_rate(
        borrower_data=data,
        loan_amount=loan_amount,
        kmeans=kmeans,
        scaler=kesonic_scaler,
        cluster_info=cluster_info
    )

    response = {
        "approved": True,
        "approval_probability": round(float(prob), 4),
        "loan_amount": round(loan_amount, 2),
        "interest_rate": round(kesonic_result["interest_rate"], 2),
        "cluster": kesonic_result["cluster_name"]
    }

    # For aproved Wait for final processing with requested loan amount
    print("✓ Approved application - Waiting for final loan amount from user")
    return jsonify(response)

# endpoint for processing final loan
@app.route("/process-loan", methods=["POST"])
def process_loan():
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['requested_loan']
    for field in required_fields:
        if field not in data:
            return jsonify({
                "success": False,
                "error": f"Missing required field: {field}"
            }), 400
    
    # Send to Google Sheets with actual requested amount if loan was approved
    if data.get('status') == 'Approved' or data.get('status') == 'Loan Processed':
        response_data = {
            "approved": True,
            "loan_amount": data.get("available_loan_limit", 0),
            "interest_rate": data.get("interest_rate", 0),
            "requested_loan": data.get("requested_loan", 0)
        }
        
        # Send to Google Sheets with final data
        success = _send_to_sheets(data, response_data, is_final=True)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Loan processed and saved to Google Sheets",
                "requested_loan": data.get("requested_loan", 0)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to save to Google Sheets"
            }), 500
    else:
        return jsonify({
            "success": False,
            "error": "Invalid status for loan processing"
        }), 400

# Google Sheets helper function
def _send_to_sheets(input_data, result_data, is_final=False):
    # Determine if approved or rejected
    is_approved = result_data.get('approved', False)
    
    # Set status
    if is_approved:
        status = "Approved"
    else:
        status = "Rejected"
    
    # Determine financial values based on approval status
    if is_approved:
        # Use actual values
        available_loan_limit = result_data.get('loan_amount', 0)
        interest_rate = result_data.get('interest_rate', 0)
        
        #  check if this is from process-loan endpoint
        if 'requested_loan' in input_data:
            requested_loan = input_data.get('requested_loan', 0)
        else:
            # Initial approval
            requested_loan = 0
    else:
        # in rejected All financial columns should be 0
        available_loan_limit = 0
        interest_rate = 0
        requested_loan = 0
    
    # Prepare complete payload
    payload = {
        'name': input_data.get('name', ''),
        'email': input_data.get('email', ''),
        'age': input_data.get('age', 0),
        'gender': input_data.get('gender', ''),
        'credit_score': input_data.get('credit_score', 0),
        'credit_history_length': input_data.get('credit_history_length', 0),
        'income': input_data.get('income', 0),
        'dti_ratio': input_data.get('dti_ratio', 0),
        'default_history': input_data.get('default_history', 0),
        'home_ownership': input_data.get('home_ownership', ''),
        'loan_purpose': input_data.get('loan_purpose', ''),
        'status': status,
        'available_loan_limit': available_loan_limit,
        'interest_rate': interest_rate,
        'requested_loan': requested_loan
    }
        
    try:
        import json
        
        # Send as raw JSON
        headers = {'Content-Type': 'application/json'}
        json_str = json.dumps(payload)
        
        resp = requests.post(
            GOOGLE_SHEETS_URL, 
            data=json_str,
            headers=headers, 
            timeout=10
        )
        
        if resp.status_code == 200:
            return True
        else:
            return False
            
    except requests.exceptions.RequestException as e:
        return False

# Run app
if __name__ == "__main__":
    app.run(debug=True)