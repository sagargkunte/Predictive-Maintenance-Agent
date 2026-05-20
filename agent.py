import requests
import pandas as pd
import json
import threading
import time
from sseclient import SSEClient
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BASE_URL = "http://localhost:3000"
FEATURES = ["temperature_C", "vibration_mm_s", "rpm", "current_A"]
MACHINES = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]
BOT_TOKEN = Your_Bot_Token_Goes_Here
CHAT_ID = Your_Chat_Id_Here


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

            # Labels: 1 for normal (running), -1 for anomaly (warning/fault)
            y = np.where(df["status"] == "running", 1, -1)

            # Train Random Forest
            print(f"[{self.machine_id}] Training Random Forest on {len(X)} samples...")
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
            self.model.fit(X, y)
            print(f"[{self.machine_id}] Model trained successfully.")

        except Exception as e:
            print(f"[{self.machine_id}] Error during training: {e}")

    def explain_anomaly(self, reading):
        """Identify which features are abnormal by z-score"""
        values = np.array([reading[f] for f in FEATURES])
        z_scores = np.abs((values - self.feature_means) / (self.feature_stds + 1e-9))

        # Find all features with a z-score > 2.0
        anomalous_indices = np.where(z_scores > 2.0)[0]

        # If no single feature exceeds the threshold, use the max one
        if len(anomalous_indices) == 0:
            anomalous_indices = [np.argmax(z_scores)]

        explanations = []
        for idx in anomalous_indices:
            feature_name = FEATURES[idx]
            val = values[idx]
            norm = self.feature_means[idx]

            if "vibration" in feature_name.lower():
                explanations.append(
                    f"Bearing wear detected (Vibration: {val:.2f} vs normal {norm:.2f})"
                )
            elif "current" in feature_name.lower():
                if val > norm:
                    explanations.append(
                        f"Motor overload due to heavy load (Current: {val:.1f}A vs normal {norm:.1f}A)"
                    )
                else:
                    explanations.append(
                        f"Motor has insufficient current (Current: {val:.1f}A vs normal {norm:.1f}A)"
                    )
            elif "temperature" in feature_name.lower():
                explanations.append(
                    f"Thermal stress or cooling failure (Temp: {val:.1f}°C vs normal {norm:.1f}°C)"
                )
            elif "rpm" in feature_name.lower():
                explanations.append(
                    f"Load or mechanical issue (RPM: {val:.0f} vs normal {norm:.0f})"
                )
            else:
                display_name = (
                    feature_name.replace("_C", "")
                    .replace("_mm_s", "")
                    .replace("_A", "")
                    .replace("temperature", "TEMP")
                    .replace("vibration", "VIB")
                    .upper()
                )
                explanations.append(f"{display_name} ({val:.1f} vs normal {norm:.1f})")

        if len(explanations) > 1:
            return f"Multiple critical issues: {', '.join(explanations)}"
        else:
            return explanations[0]

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
                    pred = self.model.predict([x])[0]  # 1 for normal, -1 for anomaly

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
                            print(
                                f"\n🚨 [{self.machine_id}] ANOMALY DETECTED! {reason}"
                            )

                            self.trigger_alert(reason, reading)
                            self.schedule_maintenance()

                            self.last_alert_time = now
                            # Clear window to prevent overlapping alerts immediately
                            self.anomaly_window.clear()

        except Exception as e:
            print(f"[{self.machine_id}] Stream disconnected or error: {e}")
            time.sleep(5)
            self.start_stream()  # Simple reconnect

    def trigger_alert(self, reason, reading):
        try:
            URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload1 = {
                "machine_id": self.machine_id,
                "reason": reason,
                "reading": reading,
            }
            payload2 = {
                "chat_id": CHAT_ID,
                "text": f"🚨 [{self.machine_id}] ANOMALY DETECTED! {reason}",
            }

            # Send Telegram notification (async, don't wait)
            try:
                requests.post(URL, json=payload2, timeout=3)
            except Exception as telegram_err:
                print(
                    f"[{self.machine_id}] Telegram notification failed (non-blocking): {telegram_err}"
                )

            # Send alert to server
            resp = requests.post(f"{BASE_URL}/alert", json=payload1, timeout=5)
            if resp.status_code == 201:
                alert_data = resp.json().get("alert", {})
                alert_id = alert_data.get("id", "UNKNOWN")
                print(f"[{self.machine_id}] ✅ System Alert Raised: {alert_id}")
            else:
                print(
                    f"[{self.machine_id}] ⚠️  Alert POST failed with status {resp.status_code}: {resp.text}"
                )
        except Exception as e:
            print(f"[{self.machine_id}] ❌ Failed to raise alert: {e}")

    # Bonus
    def schedule_maintenance(self):
        try:
            payload = {"machine_id": self.machine_id}
            resp = requests.post(f"{BASE_URL}/schedule-maintenance", json=payload)
            if resp.status_code == 201:
                booking = resp.json().get("booking", {})
                print(
                    f"[{self.machine_id}] Auto-scheduled Maintenance: {booking.get('slot')}"
                )
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
