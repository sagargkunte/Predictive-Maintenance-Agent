import requests
import pandas as pd
import json
import threading
import time
from sseclient import SSEClient
from sklearn.ensemble import IsolationForest
import numpy as np

BASE_URL = "http://localhost:3000"
FEATURES = ["temperature_C", "vibration_mm_s", "rpm", "current_A"]
MACHINES = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]

class MaintenanceAgent:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.model = None
        self.feature_means = None
        self.feature_stds = None
        self.anomaly_window = []  # sliding window of last 10 predictions
        self.last_alert_time = 0
        self.alert_cooldown = 300  # seconds

    def train(self):
        print(f"[{self.machine_id}] Fetching historical data for training...")
        try:
            resp = requests.get(f"{BASE_URL}/history/{self.machine_id}")
            resp.raise_for_status()
            data = resp.json()
            readings = data.get("readings", [])
            
            if not readings:
                print(f"[{self.machine_id}] No historical data found.")
                return

            df = pd.DataFrame(readings)
            X = df[FEATURES].values
            
            # Simple scaling factors for explanation
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)

            # Train Isolation Forest
            print(f"[{self.machine_id}] Training Isolation Forest on {len(X)} samples...")
            self.model = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
            self.model.fit(X)
            print(f"[{self.machine_id}] Model trained successfully.")

        except Exception as e:
            print(f"[{self.machine_id}] Error during training: {e}")

    def explain_anomaly(self, reading):
        """Identify which feature is most abnormal by z-score"""
        values = np.array([reading[f] for f in FEATURES])
        z_scores = np.abs((values - self.feature_means) / (self.feature_stds + 1e-9))
        max_idx = np.argmax(z_scores)
        feature_name = FEATURES[max_idx]
        val = values[max_idx]
        return f"High deviation detected in {feature_name} (Value: {val:.2f}, Normal: {self.feature_means[max_idx]:.2f})"

    def start_stream(self):
        url = f"{BASE_URL}/stream/{self.machine_id}"
        print(f"[{self.machine_id}] Connecting to live stream: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            client = SSEClient(response)
            for event in client.events():
                if not event.data:
                    continue
                
                # Parse JSON
                try:
                    reading = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                
                # Extract features
                x = [reading[f] for f in FEATURES]
                
                # Predict
                if self.model is not None:
                    # reshape because predict expects a 2d array
                    pred = self.model.predict([x])[0] # 1 for normal, -1 for anomaly
                    score = self.model.score_samples([x])[0]
                    
                    # Normal score is usually >0 (up to 0.5ish), anomalies are < 0.
                    # We map: score of 0.2 -> 0% risk, score of -0.3 -> 100% risk.
                    risk_value = (0.2 - score) * 200
                    risk_score = max(0, min(100, int(risk_value)))
                    
                    if risk_score > 80:
                        severity = "critical"
                    elif risk_score > 50:
                        severity = "warning"
                    else:
                        severity = "running"
                        
                    # Push to backend UI
                    try:
                        requests.post(f"{BASE_URL}/predict/{self.machine_id}", json={
                            "risk_score": risk_score,
                            "severity": severity
                        }, timeout=2)
                    except:
                        pass
                    
                    self.anomaly_window.append(pred)
                    if len(self.anomaly_window) > 10:
                        self.anomaly_window.pop(0)

                    # Denoiser: if >= 6 of last 10 readings are anomalous (-1), we trigger an alert
                    recent_anomalies = self.anomaly_window.count(-1)
                    
                    if recent_anomalies >= 6:
                        # Check cooldown
                        now = time.time()
                        if now - self.last_alert_time > self.alert_cooldown:
                            explanation = self.explain_anomaly(reading)
                            reason = f"Anomaly Threshold Exceeded. {explanation}"
                            print(f"\n🚨 [{self.machine_id}] ANOMALY DETECTED! {reason}")
                            
                            self.trigger_alert(reason, reading)
                            self.schedule_maintenance()
                            
                            self.last_alert_time = now
                            # Clear window to prevent overlapping alerts immediately
                            self.anomaly_window.clear()

        except Exception as e:
            print(f"[{self.machine_id}] Stream disconnected or error: {e}")
            time.sleep(5)
            self.start_stream() # Simple reconnect

    def trigger_alert(self, reason, reading):
        try:
            payload = {
                "machine_id": self.machine_id,
                "reason": reason,
                "reading": reading
            }
            resp = requests.post(f"{BASE_URL}/alert", json=payload)
            if resp.status_code == 201:
                alert_data = resp.json().get('alert', {})
                print(f"[{self.machine_id}] System Alert Raised: {alert_data.get('id')}")
        except Exception as e:
            print(f"[{self.machine_id}] Failed to raise alert: {e}")

    def schedule_maintenance(self):
        try:
            payload = {"machine_id": self.machine_id}
            resp = requests.post(f"{BASE_URL}/schedule-maintenance", json=payload)
            if resp.status_code == 201:
                booking = resp.json().get('booking', {})
                print(f"[{self.machine_id}] Auto-scheduled Maintenance: {booking.get('slot')}")
        except Exception as e:
            print(f"[{self.machine_id}] Failed to schedule maintenance: {e}")


def main():
    agents = []
    
    # 1. Initialize and train
    for m in MACHINES:
        agent = MaintenanceAgent(m)
        agent.train()
        agents.append(agent)

    print("\n🚀 All agents trained. Starting live monitoring...")

    # 2. Start streaming threads
    threads = []
    for agent in agents:
        t = threading.Thread(target=agent.start_stream, daemon=True)
        t.start()
        threads.append(t)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating ML Agent.")

if __name__ == "__main__":
    main()
