def validate_input(data):
    required_fields = [
        "credit_score",
        "credit_history_length",
        "income",
        "dti_ratio",
        "default_history",
        "home_ownership",
        "loan_purpose"
    ]

    for field in required_fields:
        if field not in data:
            return f"Missing field: {field}"

    if not (300 <= data["credit_score"] <= 900):
        return "credit_score out of range"

    if data["income"] <= 0:
        return "income must be positive"

    if not (0 <= data["dti_ratio"] <= 1):
        return "dti_ratio must be between 0 and 1"

    if data["default_history"] not in [0, 1]:
        return "default_history must be 0 or 1"

    return None
