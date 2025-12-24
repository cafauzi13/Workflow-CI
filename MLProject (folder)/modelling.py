import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Load data hasil preprocessing
    data = pd.read_csv('titanic_preprocessed.csv')
    
    # 2. Split fitur dan target
    X = data.drop('survived', axis=1)
    y = data['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Aktifkan Autolog MLflow
    mlflow.set_experiment("Titanic_Survival_Prediction")
    mlflow.autolog()

    with mlflow.start_run(run_name="Baseline_RandomForest"):
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