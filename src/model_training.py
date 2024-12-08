import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest Regressor for sales forecasting
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing target
    
    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained model
    """
    # Start MLflow tracking
    with mlflow.start_run():
        # Define model hyperparameters
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
        
        # Log model parameters
        mlflow.log_params(model_params)
        
        # Train the model
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "r2_score": r2
        })
        
        # Log the model
        mlflow.sklearn.log_model(model, "sales_forecast_model")
        
        return model

def save_model(model, filepath):
    """
    Save the trained model
    
    Args:
        model (sklearn.ensemble.RandomForestRegressor): Trained model
        filepath (str): Path to save the model
    """
    import joblib
    joblib.dump(model, filepath)