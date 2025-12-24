import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Load data hasil preprocessing (Pastikan nama file sesuai di folder MLProject)
    try:
        data = pd.read_csv('titanic_clean.csv') 
    except FileNotFoundError:
        data = pd.read_csv('titanic_preprocessed.csv')
    
    # 2. Split fitur dan target
    X = data.drop('survived', axis=1)
    y = data['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Aktifkan Autolog MLflow
    mlflow.set_experiment("Titanic_Survival_Prediction")
    mlflow.autolog()

    # LOGIKA REVISI: Cek apakah sudah ada run yang aktif dari MLflow CLI/GitHub Actions
    active_run = mlflow.active_run()

    if active_run:
        # Jika dijalankan via 'mlflow run' (GitHub Actions), tidak perlu start_run lagi
        print(f"Menggunakan Active Run ID: {active_run.info.run_id}")
        execute_training(X_train, X_test, y_train, y_test)
    else:
        # Jika dijalankan manual 'python modelling.py' di lokal
        with mlflow.start_run(run_name="Baseline_RandomForest"):
            execute_training(X_train, X_test, y_train, y_test)

def execute_training(X_train, X_test, y_train, y_test):
    # 4. Inisialisasi dan latih model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Prediksi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train_model()