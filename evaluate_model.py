import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

BASE_URL = "http://localhost:3000"
FEATURES = ["temperature_C", "vibration_mm_s", "rpm", "current_A"]
MACHINES = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

def evaluate():
    print("Evaluating Isolation Forest model on historical data...")
    try:
        # Check if server is running
        requests.get(f"{BASE_URL}/machines", timeout=5)
    except Exception:
        print("Error: Could not connect to the backend server. Please make sure 'node server.js' is running.")
        return

    for machine_id in MACHINES:
        print(f"\n--- Machine: {machine_id} ---")
        try:
            resp = requests.get(f"{BASE_URL}/history/{machine_id}")
            resp.raise_for_status()
            data = resp.json()
            readings = data.get("readings", [])
            
            if not readings:
                print(f"No historical data found for {machine_id}.")
                continue

            df = pd.DataFrame(readings)
            X = df[FEATURES].values
            
            # Ground truth: 1 for normal, -1 for anomaly
            # "running" -> 1, "warning" or "fault" -> -1
            y = np.where(df["status"] == "running", 1, -1)
            
            # Split data to evaluate properly
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest (same as agent.py)
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Metrics
            # We want to measure how well it detects anomalies (-1)
            # We'll use pos_label=-1 for precision, recall, f1
            precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Total Test Samples: {len(y_test)}")
            print(f"True Anomalies in Test: {np.sum(y_test == -1)}")
            print(f"Predicted Anomalies in Test: {np.sum(y_pred == -1)}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f} (When model predicts anomaly, how often is it actually an anomaly?)")
            print(f"Recall:    {recall:.4f} (Out of all true anomalies, how many did the model detect?)")
            print(f"F1-Score:  {f1:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {machine_id}: {e}")

if __name__ == "__main__":
    evaluate()
