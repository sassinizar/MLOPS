import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

def load_data(file_path):
    """
    Load sales data from CSV file
    
    Args:
        file_path (str): Path to the sales data CSV
    
    Returns:
        pandas.DataFrame: Loaded sales data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_param("data_load_error", str(e))
        raise

def preprocess_data(data):
    """
    Preprocess sales data for machine learning
    
    Args:
        data (pandas.DataFrame): Raw sales data
    
    Returns:
        tuple: Preprocessed features and target
    """
    # Feature engineering
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    data['quarter'] = data['date'].dt.quarter
    
    # Select features and target
    features = ['month', 'day_of_week', 'quarter', 'previous_sales', 'promotional_activity']
    target = 'sales'
    
    # Log preprocessing details
    mlflow.log_params({
        "features": features,
        "target": target,
        "total_samples": len(data)
    })
    
    # Prepare features and target
    X = data[features]
    y = data[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler