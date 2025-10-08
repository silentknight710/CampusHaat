# src/analytics/customer_segmentation.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import joblib
import logging

class CustomerSegmentation:
    """
    Customer segmentation using K-means clustering based on ordering frequency and spending patterns
    """
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the segmentation process"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for customer segmentation
        
        Args:
            df: Raw transaction data
            
        Returns:
            DataFrame with customer-level features
        """
        self.logger.info("Preparing customer features for segmentation...")
        
        # Calculate customer-level metrics
        customer_features = df.groupby('customer_id').agg({
            'order_value': ['sum', 'mean', 'count'],
            'delivery_time': 'mean',
            'order_date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
        
        # Calculate additional features
        customer_features['total_spent'] = customer_features['order_value_sum']
        customer_features['avg_order_value'] = customer_features['order_value_mean']
        customer_features['order_frequency'] = customer_features['order_value_count']
        customer_features['avg_delivery_time'] = customer_features['delivery_time_mean']
        
        # Calculate customer lifetime (days)
        customer_features['customer_lifetime'] = (
            pd.to_datetime(customer_features['order_date_max']) - 
            pd.to_datetime(customer_features['order_date_min'])
        ).dt.days + 1
        
        # Calculate orders per day
        customer_features['orders_per_day'] = (
            customer_features['order_frequency'] / customer_features['customer_lifetime']
        ).fillna(customer_features['order_frequency'])
        
        # Select final features for clustering
        features_for_clustering = [
            'total_spent', 'order_frequency', 'avg_order_value', 
            'orders_per_day', 'avg_delivery_time'
        ]
        
        return customer_features[features_for_clustering].fillna(0)
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        self.logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Find optimal k using silhouette score
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def perform_segmentation(self, df: pd.DataFrame, optimize_clusters: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform customer segmentation
        
        Args:
            df: Raw transaction data
            optimize_clusters: Whether to optimize number of clusters
            
        Returns:
            Tuple of (segmented_customers, segment_profiles)
        """
        self.logger.info("Starting customer segmentation process...")
        
        # Prepare features
        customer_features = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(customer_features)
        
        # Optimize clusters if requested
        if optimize_clusters:
            optimal_k = self.find_optimal_clusters(pd.DataFrame(X_scaled))
            self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to customer features
        customer_features['segment'] = cluster_labels
        
        # Create segment profiles
        segment_profiles = self._create_segment_profiles(customer_features)
        
        self.logger.info(f"Segmentation completed. Found {len(segment_profiles)} segments.")
        
        return customer_features, segment_profiles
    
    def _create_segment_profiles(self, segmented_data: pd.DataFrame) -> Dict:
        """
        Create detailed profiles for each customer segment
        
        Args:
            segmented_data: Customer data with segment labels
            
        Returns:
            Dictionary with segment profiles
        """
        profiles = {}
        
        for segment in segmented_data['segment'].unique():
            segment_data = segmented_data[segmented_data['segment'] == segment]
            
            profiles[f'Segment_{segment}'] = {
                'size': len(segment_data),
                'avg_total_spent': segment_data['total_spent'].mean(),
                'avg_order_frequency': segment_data['order_frequency'].mean(),
                'avg_order_value': segment_data['avg_order_value'].mean(),
                'avg_orders_per_day': segment_data['orders_per_day'].mean(),
                'avg_delivery_time': segment_data['avg_delivery_time'].mean(),
                'characteristics': self._determine_segment_characteristics(segment_data)
            }
        
        return profiles
    
    def _determine_segment_characteristics(self, segment_data: pd.DataFrame) -> str:
        """
        Determine characteristics of a customer segment
        
        Args:
            segment_data: Data for a specific segment
            
        Returns:
            String describing segment characteristics
        """
        avg_spent = segment_data['total_spent'].mean()
        avg_frequency = segment_data['order_frequency'].mean()
        avg_order_value = segment_data['avg_order_value'].mean()
        
        if avg_spent > 2000 and avg_frequency > 15:
            return "High-Value Frequent Customers"
        elif avg_spent > 1000 and avg_frequency > 10:
            return "Medium-Value Regular Customers"
        elif avg_frequency > 20:
            return "High-Frequency Low-Spend Customers"
        elif avg_order_value > 150:
            return "High-Ticket Occasional Customers"
        else:
            return "Low-Value Infrequent Customers"
    
    def save_model(self, filepath: str):
        """Save the trained segmentation model"""
        model_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained segmentation model"""
        model_data = joblib.load(filepath)
        self.kmeans = model_data['kmeans']
        self.scaler = model_data['scaler']
        self.n_clusters = model_data['n_clusters']
        self.logger.info(f"Model loaded from {filepath}")
    
    def visualize_segments(self, segmented_data: pd.DataFrame, save_path: str = None):
        """
        Create visualizations for customer segments
        
        Args:
            segmented_data: Customer data with segment labels
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Total Spent vs Order Frequency
        axes[0, 0].scatter(segmented_data['total_spent'], segmented_data['order_frequency'], 
                          c=segmented_data['segment'], cmap='viridis', alpha=0.6)
        axes[0, 0].set_xlabel('Total Spent')
        axes[0, 0].set_ylabel('Order Frequency')
        axes[0, 0].set_title('Customer Segments: Spending vs Frequency')
        
        # Scatter plot: Average Order Value vs Orders per Day
        axes[0, 1].scatter(segmented_data['avg_order_value'], segmented_data['orders_per_day'], 
                          c=segmented_data['segment'], cmap='viridis', alpha=0.6)
        axes[0, 1].set_xlabel('Average Order Value')
        axes[0, 1].set_ylabel('Orders per Day')
        axes[0, 1].set_title('Customer Segments: Order Value vs Daily Frequency')
        
        # Segment size distribution
        segment_counts = segmented_data['segment'].value_counts().sort_index()
        axes[1, 0].bar(segment_counts.index, segment_counts.values, color='lightblue')
        axes[1, 0].set_xlabel('Segment')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Customer Distribution by Segment')
        
        # Box plot: Total Spent by Segment
        segmented_data.boxplot(column='total_spent', by='segment', ax=axes[1, 1])
        axes[1, 1].set_title('Total Spending Distribution by Segment')
        axes[1, 1].set_xlabel('Segment')
        axes[1, 1].set_ylabel('Total Spent')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('../data/processed/cleaned_transactions.csv')
    
    # Initialize segmentation
    segmenter = CustomerSegmentation()
    
    # Perform segmentation
    segmented_customers, profiles = segmenter.perform_segmentation(df)
    
    # Save results
    segmented_customers.to_csv('../data/processed/customer_segments.csv', index=True)
    
    # Save model
    segmenter.save_model('../models/customer_segmentation_model.pkl')
    
    # Create visualizations
    segmenter.visualize_segments(segmented_customers, '../dashboard/static/images/customer_segments.png')
    
    # Print segment profiles
    for segment, profile in profiles.items():
        print(f"\n{segment}:")
        print(f"  Size: {profile['size']} customers")
        print(f"  Characteristics: {profile['characteristics']}")
        print(f"  Avg Total Spent: ₹{profile['avg_total_spent']:.2f}")
        print(f"  Avg Order Frequency: {profile['avg_order_frequency']:.1f} orders")
        print(f"  Avg Order Value: ₹{profile['avg_order_value']:.2f}")