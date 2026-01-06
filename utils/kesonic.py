import numpy as np
import pandas as pd


def calculate_kesonic_rate(borrower_data, loan_amount, kmeans, scaler, cluster_info):
    kesonic_features = cluster_info["feature_names"]
    cluster_names = cluster_info["cluster_names"]
    premium_mapping = cluster_info["premium_mapping"]
    max_distance = cluster_info["max_distance"]
    kesonia_base = cluster_info["kesonia_base"]
    margin = cluster_info["margin"]

    data = borrower_data.copy()
    data["loan_amount"] = loan_amount
    data["employment_length"] = 5

    df = pd.DataFrame([data])

    df = pd.get_dummies(
        df,
        columns=["home_ownership", "loan_purpose"],
        drop_first=True
    )

    for col in kesonic_features:
        if col not in df.columns:
            df[col] = 0

    df = df[kesonic_features]
    scaled = scaler.transform(df)

    cluster = kmeans.predict(scaled)[0]
    cluster_name = cluster_names.get(cluster, "Standard")

    centroid = kmeans.cluster_centers_[cluster]
    distance = np.linalg.norm(scaled - centroid)
    premium_adjustment = (distance / max_distance) * 10

    base_premium = premium_mapping.get(cluster_name, 1.0)
    final_rate = base_premium + premium_adjustment + margin
    interest_rate = kesonia_base + final_rate

    return {
        "cluster": int(cluster),
        "cluster_name": cluster_name,
        "interest_rate": float(interest_rate)
    }
