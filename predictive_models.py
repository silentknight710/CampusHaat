# src/ml_models/predictive_models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
import joblib
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeliveryTimePredictor:
    """
    Machine Learning model to predict delivery times based on various factors
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_version = "1.0"
        self.logger = self._setup_logger()
        self.is_trained = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the model"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for delivery time prediction
        
        Args:
            df: Raw transaction data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Preparing features for delivery time prediction...")
        
        features_df = df.copy()
        
        # Time-based features
        features_df['order_date'] = pd.to_datetime(features_df['order_date'])
        features_df['hour_of_day'] = features_df['order_date'].dt.hour
        features_df['day_of_week'] = features_df['order_date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Peak hour identification
        features_df['is_peak_hour'] = features_df['hour_of_day'].isin([12, 13, 19, 20, 21]).astype(int)
        
        # Distance estimation (simplified - in real scenario, use actual coordinates)
        np.random.seed(42)
        features_df['estimated_distance_km'] = np.random.uniform(0.5, 8.0, len(features_df))
        
        # Restaurant-based features
        restaurant_stats = df.groupby('restaurant_id').agg({
            'delivery_time': ['mean', 'std', 'count'],
            'order_value': 'mean'
        }).round(2)
        
        restaurant_stats.columns = ['_'.join(col) for col in restaurant_stats.columns]
        restaurant_stats = restaurant_stats.add_prefix('restaurant_')
        
        features_df = features_df.merge(
            restaurant_stats, 
            left_on='restaurant_id', 
            right_index=True, 
            how='left'
        )
        
        # City-based features
        city_stats = df.groupby('city').agg({
            'delivery_time': ['mean', 'std'],
            'order_value': 'mean'
        }).round(2)
        
        city_stats.columns = ['_'.join(col) for col in city_stats.columns]
        city_stats = city_stats.add_prefix('city_')
        
        features_df = features_df.merge(
            city_stats,
            left_on='city',
            right_index=True,
            how='left'
        )
        
        # Weather impact (simplified - in real scenario, use weather API)
        weather_conditions = ['Clear', 'Rain', 'Cloudy', 'Storm']
        features_df['weather_condition'] = np.random.choice(weather_conditions, len(features_df))
        
        # Order complexity features
        features_df['order_complexity'] = pd.cut(
            features_df['order_value'], 
            bins=[0, 100, 300, 500, float('inf')], 
            labels=['Simple', 'Medium', 'Complex', 'Very Complex']
        ).astype(str)
        
        return features_df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding
        
        Args:
            df: DataFrame with features
            fit: Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            DataFrame with encoded features
        """
        categorical_columns = ['city', 'cuisine_type', 'weather_condition', 'order_complexity']
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df_encoded[f'{col}_encoded'] = df_encoded[col].astype(str).map(
                            lambda x: self.label_encoders[col].transform([x])[0] 
                            if x in self.label_encoders[col].classes_ else -1
                        )
                    else:
                        df_encoded[f'{col}_encoded'] = 0
        
        return df_encoded
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for model training
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of selected feature names
        """
        feature_columns = [
            'order_value', 'estimated_distance_km', 'hour_of_day', 'day_of_week',
            'is_weekend', 'is_peak_hour', 'restaurant_delivery_time_mean',
            'restaurant_delivery_time_std', 'restaurant_delivery_time_count',
            'restaurant_order_value_mean', 'city_delivery_time_mean',
            'city_delivery_time_std', 'city_order_value_mean',
            'city_encoded', 'cuisine_type_encoded', 'weather_condition_encoded',
            'order_complexity_encoded'
        ]
        
        # Return only columns that exist in the dataframe
        return [col for col in feature_columns if col in df.columns]
    
    def train(self, df: pd.DataFrame, target_column: str = 'delivery_time') -> Dict:
        """
        Train the delivery time prediction model
        
        Args:
            df: Training data
            target_column: Name of target column
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting model training...")
        
        # Prepare features
        features_df = self.prepare_features(df)
        features_df = self.encode_categorical_features(features_df, fit=True)
        
        # Select features and target
        feature_columns = self.select_features(features_df)
        X = features_df[feature_columns].fillna(0)
        y = features_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        self.logger.info("Performing hyperparameter tuning...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Train best model
        self.model = grid_search.best_estimator_
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # Feature importance
        self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_absolute_error'
        )
        
        self.is_trained = True
        
        metrics = {
            'train_mae': round(train_mae, 2),
            'test_mae': round(test_mae, 2),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'cv_mae_mean': round(-cv_scores.mean(), 2),
            'cv_mae_std': round(cv_scores.std(), 2),
            'best_params': grid_search.best_params_,
            'feature_importance': self.feature_importance
        }
        
        self.logger.info(f"Model training completed. Test MAE: {test_mae:.2f} minutes")
        return metrics
    
    def predict(self, features: Dict) -> float:
        """
        Predict delivery time for a single order
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Predicted delivery time in minutes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Prepare features
        df = self.encode_categorical_features(df, fit=False)
        
        # Select and scale features
        feature_columns = self.select_features(df)
        X = df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        return max(prediction, 10)  # Minimum 10 minutes delivery time
    
    def get_prediction_interval(self, features: Dict, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Get prediction interval for delivery time
        
        Args:
            features: Dictionary of feature values
            confidence: Confidence level for interval
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        prediction = self.predict(features)
        
        # Simplified prediction interval (in practice, use quantile regression)
        std_error = 5.0  # Estimated standard error
        z_score = 1.96 if confidence == 0.95 else 2.58  # For 95% or 99% confidence
        
        margin_of_error = z_score * std_error
        lower_bound = max(prediction - margin_of_error, 10)
        upper_bound = prediction + margin_of_error
        
        return (lower_bound, upper_bound)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        self.model_version = model_data['model_version']
        self.is_trained = model_data['is_trained']
        self.logger.info(f"Model loaded from {filepath}")


class ChurnPredictor:
    """
    Machine Learning model to predict customer churn
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_version = "1.0"
        self.logger = self._setup_logger()
        self.is_trained = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the model"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def prepare_churn_features(self, df: pd.DataFrame, prediction_window_days: int = 30) -> pd.DataFrame:
        """
        Prepare features for churn prediction
        
        Args:
            df: Transaction data
            prediction_window_days: Days ahead to predict churn
            
        Returns:
            DataFrame with customer-level features and churn labels
        """
        self.logger.info("Preparing features for churn prediction...")
        
        # Calculate observation period
        max_date = pd.to_datetime(df['order_date']).max()
        observation_end = max_date - timedelta(days=prediction_window_days)
        observation_start = observation_end - timedelta(days=90)  # 90-day observation window
        
        # Filter data for observation period
        df_obs = df[
            (pd.to_datetime(df['order_date']) >= observation_start) &
            (pd.to_datetime(df['order_date']) <= observation_end)
        ].copy()
        
        # Calculate customer features during observation period
        customer_features = df_obs.groupby('customer_id').agg({
            'order_id': 'count',
            'order_value': ['sum', 'mean', 'std'],
            'delivery_time': ['mean', 'std'],
            'order_date': ['min', 'max', 'count']
        }).round(2)
        
        # Flatten column names
        customer_features.columns = ['_'.join(col) for col in customer_features.columns]
        
        # Calculate additional features
        customer_features['total_orders'] = customer_features['order_id_count']
        customer_features['total_spent'] = customer_features['order_value_sum']
        customer_features['avg_order_value'] = customer_features['order_value_mean']
        customer_features['order_value_variability'] = customer_features['order_value_std'].fillna(0)
        customer_features['avg_delivery_time'] = customer_features['delivery_time_mean']
        customer_features['delivery_consistency'] = customer_features['delivery_time_std'].fillna(0)
        
        # Calculate time-based features
        customer_features['days_active'] = (
            pd.to_datetime(customer_features['order_date_max']) - 
            pd.to_datetime(customer_features['order_date_min'])
        ).dt.days + 1
        
        customer_features['order_frequency'] = (
            customer_features['total_orders'] / customer_features['days_active']
        ).fillna(customer_features['total_orders'])
        
        # Calculate days since last order (as of observation end)
        customer_features['days_since_last_order'] = (
            observation_end - pd.to_datetime(customer_features['order_date_max'])
        ).dt.days
        
        # Determine churn label (customer didn't order in prediction window)
        future_customers = set(df[
            pd.to_datetime(df['order_date']) > observation_end
        ]['customer_id'].unique())
        
        customer_features['churned'] = ~customer_features.index.isin(future_customers)
        
        # Add categorical features from most recent orders
        recent_orders = df_obs.sort_values('order_date').groupby('customer_id').tail(1)
        categorical_features = recent_orders[['customer_id', 'city', 'cuisine_type']].set_index('customer_id')
        
        customer_features = customer_features.join(categorical_features, how='left')
        
        # Select final features
        feature_columns = [
            'total_orders', 'total_spent', 'avg_order_value', 'order_value_variability',
            'avg_delivery_time', 'delivery_consistency', 'days_active', 'order_frequency',
            'days_since_last_order', 'city', 'cuisine_type'
        ]
        
        return customer_features[feature_columns + ['churned']].fillna(0)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the churn prediction model
        
        Args:
            df: Transaction data
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info("Starting churn prediction model training...")
        
        # Prepare features
        features_df = self.prepare_churn_features(df)
        
        # Encode categorical features
        categorical_columns = ['city', 'cuisine_type']
        for col in categorical_columns:
            if col in features_df.columns:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    features_df[col].astype(str)
                )
        
        # Select features and target
        feature_columns = [
            'total_orders', 'total_spent', 'avg_order_value', 'order_value_variability',
            'avg_delivery_time', 'delivery_consistency', 'days_active', 'order_frequency',
            'days_since_last_order', 'city_encoded', 'cuisine_type_encoded'
        ]
        
        X = features_df[feature_columns]
        y = features_df['churned'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        y_pred_proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Feature importance
        self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        self.is_trained = True
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'train_accuracy': round(accuracy_score(y_train, y_pred_train), 3),
            'test_accuracy': round(accuracy_score(y_test, y_pred_test), 3),
            'precision': round(precision_score(y_test, y_pred_test), 3),
            'recall': round(recall_score(y_test, y_pred_test), 3),
            'f1_score': round(f1_score(y_test, y_pred_test), 3),
            'roc_auc': round(roc_auc_score(y_test, y_pred_proba_test), 3),
            'feature_importance': self.feature_importance
        }
        
        self.logger.info(f"Churn model training completed. Test AUC: {metrics['roc_auc']}")
        return metrics
    
    def predict_churn_probability(self, customer_features: Dict) -> float:
        """
        Predict churn probability for a customer
        
        Args:
            customer_features: Dictionary of customer features
            
        Returns:
            Churn probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_features])
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[f'{col}_encoded'] = encoder.transform(df[col].astype(str))
        
        # Select features
        feature_columns = [
            'total_orders', 'total_spent', 'avg_order_value', 'order_value_variability',
            'avg_delivery_time', 'delivery_consistency', 'days_active', 'order_frequency',
            'days_since_last_order', 'city_encoded', 'cuisine_type_encoded'
        ]
        
        X = df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict probability
        churn_probability = self.model.predict_proba(X_scaled)[0, 1]
        return round(churn_probability, 3)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Churn model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_importance = model_data['feature_importance']
        self.model_version = model_data['model_version']
        self.is_trained = model_data['is_trained']
        self.logger.info(f"Churn model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('../data/processed/cleaned_transactions.csv')
    
    # Test Delivery Time Predictor
    delivery_predictor = DeliveryTimePredictor()
    delivery_metrics = delivery_predictor.train(df)
    
    print("Delivery Time Prediction Metrics:")
    for key, value in delivery_metrics.items():
        if key != 'feature_importance':
            print(f"  {key}: {value}")
    
    # Test prediction
    sample_features = {
        'order_value': 250.0,
        'estimated_distance_km': 3.5,
        'hour_of_day': 19,
        'day_of_week': 1,
        'is_weekend': 0,
        'is_peak_hour': 1,
        'city': 'Mumbai',
        'cuisine_type': 'Indian',
        'weather_condition': 'Clear',
        'restaurant_id': 'REST_001'
    }
    
    predicted_time = delivery_predictor.predict(sample_features)
    print(f"\nSample prediction: {predicted_time:.1f} minutes")
    
    # Save models
    delivery_predictor.save_model('../models/delivery_time_model.pkl')
    
    # Test Churn Predictor
    churn_predictor = ChurnPredictor()
    churn_metrics = churn_predictor.train(df)
    
    print("\nChurn Prediction Metrics:")
    for key, value in churn_metrics.items():
        if key != 'feature_importance':
            print(f"  {key}: {value}")
    
    churn_predictor.save_model('../models/churn_prediction_model.pkl')