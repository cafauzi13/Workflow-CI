import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Load data (Handle 2 kemungkinan nama file)
    try:
        data = pd.read_csv('titanic_clean.csv')
    except:
        data = pd.read_csv('titanic_preprocessed.csv')
    
    X = data.drop('survived', axis=1)
    y = data['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Aktifkan Autolog
    mlflow.autolog()

    # --- LOGIKA PENYELAMAT ---
    # Jika dijalankan lewat 'mlflow run' (GitHub Actions), active_run() tidak akan None
    if mlflow.active_run():
        print("Mendeteksi Active Run dari MLflow Runner...")
        execute_logic(X_train, X_test, y_train, y_test)
    else:
        # Jika dijalankan manual 'python modelling.py' di lokal
        print("Menjalankan sebagai script standalone...")
        mlflow.set_experiment("Titanic_Survival_Prediction")
        with mlflow.start_run(run_name="Baseline_RandomForest"):
            execute_logic(X_train, X_test, y_train, y_test)

def execute_logic(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Akurasi Model: {acc:.4f}")

if __name__ == "__main__":
    train_model()