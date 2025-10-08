# src/api/app.py

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional
import logging
from functools import wraps
import jwt
import os
from werkzeug.security import check_password_hash

# Import custom modules
import sys
sys.path.append('..')
from analytics.customer_segmentation import CustomerSegmentation
from analytics.retention_analysis import RetentionAnalysis
from analytics.operational_analysis import OperationalAnalysis
from ml_models.predictive_models import DeliveryTimePredictor, ChurnPredictor

class FoodDeliveryAPI:
    """
    RESTful API for Food Delivery Analytics Platform
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
        self.api = Api(self.app)
        CORS(self.app)
        
        # Initialize analytics modules
        self.customer_segmentation = CustomerSegmentation()
        self.retention_analysis = RetentionAnalysis()
        self.operational_analysis = OperationalAnalysis()
        self.delivery_predictor = DeliveryTimePredictor()
        self.churn_predictor = ChurnPredictor()
        
        # Setup logging
        self.setup_logging()
        
        # Register API endpoints
        self.register_endpoints()
        
        # Load models
        self.load_models()
    
    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('api.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            self.customer_segmentation.load_model('../models/customer_segmentation_model.pkl')
            self.delivery_predictor.load_model('../models/delivery_time_model.pkl')
            self.churn_predictor.load_model('../models/churn_prediction_model.pkl')
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load some models: {str(e)}")
    
    def token_required(self, f):
        """Decorator for JWT authentication"""
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'message': 'Token is missing'}), 401
            
            try:
                token = token.split(' ')[1]  # Remove 'Bearer ' prefix
                data = jwt.decode(token, self.app.config['SECRET_KEY'], algorithms=['HS256'])
            except:
                return jsonify({'message': 'Token is invalid'}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    def get_db_connection(self):
        """Get database connection"""
        conn = sqlite3.connect('../data/food_delivery.db')
        conn.row_factory = sqlite3.Row
        return conn
    
    def register_endpoints(self):
        """Register all API endpoints"""
        
        # Analytics endpoints
        self.api.add_resource(CustomerSegmentsAPI, '/api/v1/analytics/segments',
                             resource_class_kwargs={'api_instance': self})
        
        self.api.add_resource(RetentionMetricsAPI, '/api/v1/analytics/retention',
                             resource_class_kwargs={'api_instance': self})
        
        self.api.add_resource(OperationalMetricsAPI, '/api/v1/analytics/operations',
                             resource_class_kwargs={'api_instance': self})
        
        # Prediction endpoints
        self.api.add_resource(DeliveryTimePredictionAPI, '/api/v1/predict/delivery-time',
                             resource_class_kwargs={'api_instance': self})
        
        self.api.add_resource(ChurnPredictionAPI, '/api/v1/predict/churn',
                             resource_class_kwargs={'api_instance': self})
        
        # Dashboard endpoints
        self.api.add_resource(DashboardMetricsAPI, '/api/v1/dashboard/metrics',
                             resource_class_kwargs={'api_instance': self})
        
        self.api.add_resource(RecommendationsAPI, '/api/v1/recommendations',
                             resource_class_kwargs={'api_instance': self})
    
    def run(self, debug=True, host='0.0.0.0', port=5000):
        """Run the Flask application"""
        self.logger.info(f"Starting Food Delivery Analytics API on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)


# API Resource Classes

class CustomerSegmentsAPI(Resource):
    """API endpoint for customer segmentation data"""
    
    def __init__(self, api_instance):
        self.api = api_instance
    
    def get(self):
        """Get customer segments data"""
        try:
            # Get query parameters
            city = request.args.get('city', 'all')
            date_from = request.args.get('date_from')
            date_to = request.args.get('date_to')
            
            # Fetch data from database
            conn = self.api.get_db_connection()
            
            query = """
                SELECT customer_id, city, order_value, delivery_time, order_date,
                       restaurant_id, cuisine_type
                FROM transactions
                WHERE 1=1
            """
            
            params = []
            if city != 'all':
                query += " AND city = ?"
                params.append(city)
            
            if date_from:
                query += " AND order_date >= ?"
                params.append(date_from)
            
            if date_to:
                query += " AND order_date <= ?"
                params.append(date_to)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return {'message': 'No data found for specified filters'}, 404
            
            # Perform segmentation
            segmented_customers, profiles = self.api.customer_segmentation.perform_segmentation(df)
            
            # Prepare response
            response_data = {
                'segments': profiles,
                'total_customers': len(segmented_customers),
                'segmentation_date': datetime.now().isoformat(),
                'filters_applied': {
                    'city': city,
                    'date_from': date_from,
                    'date_to': date_to
                }
            }
            
            return response_data, 200
            
        except Exception as e:
            self.api.logger.error(f"Error in customer segmentation API: {str(e)}")
            return {'error': 'Internal server error'}, 500


class RetentionMetricsAPI(Resource):
    """API endpoint for customer retention metrics"""
    
    def __init__(self, api_instance):
        self.api = api_instance
    
    def get(self):
        """Get customer retention metrics"""
        try:
            # Get query parameters
            cohort_period = request.args.get('cohort_period', 'monthly')  # weekly, monthly
            city = request.args.get('city', 'all')
            
            # Fetch transaction data
            conn = self.api.get_db_connection()
            
            query = """
                SELECT customer_id, order_date, order_value, city
                FROM transactions
                ORDER BY customer_id, order_date
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if city != 'all':
                df = df[df['city'] == city]
            
            # Calculate retention metrics
            retention_data = self.api.retention_analysis.calculate_cohort_retention(
                df, period=cohort_period
            )
            
            # Calculate additional metrics
            churn_rate = self.api.retention_analysis.calculate_churn_rate(df)
            customer_lifetime_value = self.api.retention_analysis.calculate_clv(df)
            
            response_data = {
                'cohort_retention': retention_data.to_dict(),
                'average_churn_rate': churn_rate,
                'average_clv': customer_lifetime_value,
                'cohort_period': cohort_period,
                'city': city,
                'analysis_date': datetime.now().isoformat()
            }
            
            return response_data, 200
            
        except Exception as e:
            self.api.logger.error(f"Error in retention metrics API: {str(e)}")
            return {'error': 'Internal server error'}, 500


class OperationalMetricsAPI(Resource):
    """API endpoint for operational metrics and KPIs"""
    
    def __init__(self, api_instance):
        self.api = api_instance
    
    def get(self):
        """Get operational metrics"""
        try:
            # Get query parameters
            date_from = request.args.get('date_from', 
                                       (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            date_to = request.args.get('date_to', datetime.now().strftime('%Y-%m-%d'))
            city = request.args.get('city', 'all')
            
            # Fetch data
            conn = self.api.get_db_connection()
            
            query = """
                SELECT order_id, customer_id, restaurant_id, city, order_value,
                       delivery_time, order_date, delivery_status, payment_method,
                       cuisine_type, peak_hour, weather_condition
                FROM transactions
                WHERE order_date BETWEEN ? AND ?
            """
            
            params = [date_from, date_to]
            if city != 'all':
                query += " AND city = ?"
                params.append(city)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Calculate operational metrics
            metrics = self.api.operational_analysis.calculate_operational_metrics(df)
            
            # Add delivery performance metrics
            delivery_metrics = self.api.operational_analysis.analyze_delivery_performance(df)
            
            # City-wise breakdown
            city_metrics = {}
            for city_name in df['city'].unique():
                city_df = df[df['city'] == city_name]
                city_metrics[city_name] = {
                    'avg_delivery_time': city_df['delivery_time'].mean(),
                    'total_orders': len(city_df),
                    'total_revenue': city_df['order_value'].sum(),
                    'avg_order_value': city_df['order_value'].mean()
                }
            
            response_data = {
                'overall_metrics': metrics,
                'delivery_performance': delivery_metrics,
                'city_breakdown': city_metrics,
                'period': {
                    'from': date_from,
                    'to': date_to
                },
                'analysis_date': datetime.now().isoformat()
            }
            
            return response_data, 200
            
        except Exception as e:
            self.api.logger.error(f"Error in operational metrics API: {str(e)}")
            return {'error': 'Internal server error'}, 500


class DeliveryTimePredictionAPI(Resource):
    """API endpoint for delivery time predictions"""
    
    def __init__(self, api_instance):
        self.api = api_instance
    
    def post(self):
        """Predict delivery time for new orders"""
        try:
            # Get request data
            data = request.get_json()
            
            required_fields = ['restaurant_id', 'customer_location', 'order_value', 
                             'cuisine_type', 'peak_hour', 'weather_condition']
            
            # Validate input
            for field in required_fields:
                if field not in data:
                    return {'error': f'Missing required field: {field}'}, 400
            
            # Prepare features for prediction
            features = {
                'restaurant_id': data['restaurant_id'],
                'distance_km': data.get('distance_km', 3.0),  # Default distance
                'order_value': data['order_value'],
                'cuisine_type': data['cuisine_type'],
                'peak_hour': data['peak_hour'],
                'weather_condition': data['weather_condition'],
                'day_of_week': datetime.now().weekday(),
                'hour_of_day': datetime.now().hour
            }
            
            # Make prediction
            predicted_time = self.api.delivery_predictor.predict(features)
            
            # Calculate confidence interval
            confidence_interval = self.api.delivery_predictor.get_prediction_interval(
                features, confidence=0.95
            )
            
            response_data = {
                'predicted_delivery_time_minutes': round(predicted_time, 1),
                'confidence_interval': {
                    'lower_bound': round(confidence_interval[0], 1),
                    'upper_bound': round(confidence_interval[1], 1)
                },
                'prediction_timestamp': datetime.now().isoformat(),
                'model_version': self.api.delivery_predictor.model_version
            }
            
            return response_data, 200
            
        except Exception as e:
            self.api.logger.error(f"Error in delivery time prediction API: {str(e)}")
            return {'error': 'Internal server error'}, 500


class DashboardMetricsAPI(Resource):
    """API endpoint for dashboard summary metrics"""
    
    def __init__(self, api_instance):
        self.api = api_instance
    
    def get(self):
        """Get dashboard summary metrics"""
        try:
            # Get current date and calculate periods
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            last_week_start = today - timedelta(days=7)
            last_month_start = today - timedelta(days=30)
            
            conn = self.api.get_db_connection()
            
            # Today's metrics
            today_query = """
                SELECT COUNT(*) as orders, SUM(order_value) as revenue,
                       AVG(delivery_time) as avg_delivery_time
                FROM transactions
                WHERE DATE(order_date) = ?
            """
            
            today_metrics = pd.read_sql_query(today_query, conn, params=[today.isoformat()])
            
            # Yesterday's metrics for comparison
            yesterday_metrics = pd.read_sql_query(today_query, conn, params=[yesterday.isoformat()])
            
            # Weekly trends
            weekly_query = """
                SELECT DATE(order_date) as date, COUNT(*) as orders, SUM(order_value) as revenue
                FROM transactions
                WHERE DATE(order_date) >= ?
                GROUP BY DATE(order_date)
                ORDER BY date
            """
            
            weekly_data = pd.read_sql_query(weekly_query, conn, params=[last_week_start.isoformat()])
            
            # Top performing cities
            city_query = """
                SELECT city, COUNT(*) as orders, SUM(order_value) as revenue,
                       AVG(delivery_time) as avg_delivery_time
                FROM transactions
                WHERE DATE(order_date) >= ?
                GROUP BY city
                ORDER BY revenue DESC
                LIMIT 5
            """
            
            city_data = pd.read_sql_query(city_query, conn, params=[last_month_start.isoformat()])
            
            conn.close()
            
            # Calculate growth rates
            def calculate_growth(current, previous):
                if previous > 0:
                    return ((current - previous) / previous) * 100
                return 0
            
            today_orders = today_metrics.iloc[0]['orders']
            yesterday_orders = yesterday_metrics.iloc[0]['orders']
            today_revenue = today_metrics.iloc[0]['revenue'] or 0
            yesterday_revenue = yesterday_metrics.iloc[0]['revenue'] or 0
            
            response_data = {
                'summary': {
                    'today_orders': int(today_orders),
                    'today_revenue': float(today_revenue),
                    'avg_delivery_time': float(today_metrics.iloc[0]['avg_delivery_time'] or 0),
                    'orders_growth': calculate_growth(today_orders, yesterday_orders),
                    'revenue_growth': calculate_growth(today_revenue, yesterday_revenue)
                },
                'weekly_trend': weekly_data.to_dict('records'),
                'top_cities': city_data.to_dict('records'),
                'last_updated': datetime.now().isoformat()
            }
            
            return response_data, 200
            
        except Exception as e:
            self.api.logger.error(f"Error in dashboard metrics API: {str(e)}")
            return {'error': 'Internal server error'}, 500


# Initialize and run the application
if __name__ == '__main__':
    api = FoodDeliveryAPI()
    api.run(debug=True)