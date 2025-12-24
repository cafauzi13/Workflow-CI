import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    # 1. Load data - Pastikan nama file ini ada di folder MLProject
    # Kamu bisa ganti ke 'titanic_preprocessed.csv' jika itu nama filemu
    df = pd.read_csv('titanic_preprocessed.csv') 
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. AKTIFKAN AUTOLOG SAJA
    # Jangan pakai set_experiment atau start_run, ini kunci biar gak error di GitHub!
    mlflow.autolog()

    # 3. Latih model langsung
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Output sederhana untuk log
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Training Selesai! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_model()