# dashboard/dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import requests
import json
import sys
import os

# Add src directory to path for imports
sys.path.append('../src')
from analytics.customer_segmentation import CustomerSegmentation
from analytics.retention_analysis import RetentionAnalysis
from analytics.operational_analysis import OperationalAnalysis

class FoodDeliveryDashboard:
    """
    Real-time dashboard for food delivery analytics
    """
    
    def __init__(self):
        self.api_base_url = "http://localhost:5000/api/v1"
        self.setup_page_config()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Food Delivery Analytics Dashboard",
            page_icon="🍔",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect('../data/food_delivery.db')
    
    def load_data(self):
        """Load data from database"""
        conn = self.get_db_connection()
        query = """
        SELECT * FROM transactions 
        WHERE order_date >= DATE('now', '-90 days')
        ORDER BY order_date DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def create_metrics_cards(self, df):
        """Create metric cards for key KPIs"""
        # Calculate today's metrics
        today = datetime.now().date()
        today_data = df[pd.to_datetime(df['order_date']).dt.date == today]
        yesterday_data = df[pd.to_datetime(df['order_date']).dt.date == (today - timedelta(days=1))]
        
        # Metrics calculation
        today_orders = len(today_data)
        today_revenue = today_data['order_value'].sum()
        today_avg_delivery = today_data['delivery_time'].mean()
        today_customers = today_data['customer_id'].nunique()
        
        yesterday_orders = len(yesterday_data)
        yesterday_revenue = yesterday_data['order_value'].sum()
        
        # Growth calculations
        orders_growth = ((today_orders - yesterday_orders) / yesterday_orders * 100) if yesterday_orders > 0 else 0
        revenue_growth = ((today_revenue - yesterday_revenue) / yesterday_revenue * 100) if yesterday_revenue > 0 else 0
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Today's Orders",
                value=f"{today_orders:,}",
                delta=f"{orders_growth:+.1f}%" if orders_growth != 0 else None
            )
        
        with col2:
            st.metric(
                label="💰 Today's Revenue",
                value=f"₹{today_revenue:,.0f}",
                delta=f"{revenue_growth:+.1f}%" if revenue_growth != 0 else None
            )
        
        with col3:
            st.metric(
                label="⏱️ Avg Delivery Time",
                value=f"{today_avg_delivery:.1f} min" if not pd.isna(today_avg_delivery) else "N/A"
            )
        
        with col4:
            st.metric(
                label="👥 Active Customers",
                value=f"{today_customers:,}"
            )
    
    def create_revenue_trend_chart(self, df):
        """Create revenue trend chart"""
        # Daily revenue for last 30 days
        df['order_date'] = pd.to_datetime(df['order_date'])
        last_30_days = df[df['order_date'] >= (datetime.now() - timedelta(days=30))]
        
        daily_revenue = last_30_days.groupby(last_30_days['order_date'].dt.date).agg({
            'order_value': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        daily_revenue.columns = ['Date', 'Revenue', 'Orders']
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Revenue and Order Trends (Last 30 Days)"]
        )
        
        # Add revenue line
        fig.add_trace(
            go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['Revenue'],
                mode='lines+markers',
                name='Revenue (₹)',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False
        )
        
        # Add orders line
        fig.add_trace(
            go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['Orders'],
                mode='lines+markers',
                name='Orders',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Orders", secondary_y=True)
        
        fig.update_layout(
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_city_performance_chart(self, df):
        """Create city performance comparison"""
        city_metrics = df.groupby('city').agg({
            'order_value': ['sum', 'mean', 'count'],
            'delivery_time': 'mean'
        }).round(2)
        
        city_metrics.columns = ['Total Revenue', 'Avg Order Value', 'Total Orders', 'Avg Delivery Time']
        city_metrics = city_metrics.reset_index()
        
        # Create bubble chart
        fig = px.scatter(
            city_metrics,
            x='Avg Delivery Time',
            y='Avg Order Value',
            size='Total Orders',
            color='Total Revenue',
            hover_name='city',
            title='City Performance: Delivery Time vs Order Value',
            labels={
                'Avg Delivery Time': 'Average Delivery Time (minutes)',
                'Avg Order Value': 'Average Order Value (₹)',
                'Total Revenue': 'Total Revenue (₹)'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_customer_segmentation_chart(self, df):
        """Create customer segmentation visualization"""
        try:
            segmenter = CustomerSegmentation()
            segmented_customers, profiles = segmenter.perform_segmentation(df)
            
            # Create segment distribution pie chart
            segment_counts = segmented_customers['segment'].value_counts()
            segment_labels = [profiles[f'Segment_{i}']['characteristics'] for i in segment_counts.index]
            
            fig = px.pie(
                values=segment_counts.values,
                names=segment_labels,
                title='Customer Segmentation Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            return fig, profiles
        
        except Exception as e:
            st.error(f"Error creating segmentation chart: {str(e)}")
            return None, None
    
    def create_delivery_performance_chart(self, df):
        """Create delivery performance analysis"""
        # Delivery time distribution by city
        fig = px.box(
            df,
            x='city',
            y='delivery_time',
            title='Delivery Time Distribution by City',
            labels={
                'delivery_time': 'Delivery Time (minutes)',
                'city': 'City'
            }
        )
        
        # Add average line
        avg_delivery_time = df['delivery_time'].mean()
        fig.add_hline(
            y=avg_delivery_time,
            line_dash="dash",
            annotation_text=f"Overall Average: {avg_delivery_time:.1f} min",
            annotation_position="top right"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_hourly_orders_heatmap(self, df):
        """Create hourly orders heatmap"""
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['hour'] = df['order_date'].dt.hour
        df['day_name'] = df['order_date'].dt.day_name()
        
        # Create pivot table for heatmap
        hourly_orders = df.groupby(['day_name', 'hour']).size().reset_index(name='order_count')
        heatmap_data = hourly_orders.pivot(index='day_name', columns='hour', values='order_count').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(day_order)
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Order Volume Heatmap by Day and Hour",
            color_continuous_scale='YlOrRd'
        )
        
        fig.update_layout(height=400)
        return fig
    
    def display_operational_insights(self, df):
        """Display operational insights"""
        st.subheader("🔍 Operational Insights")
        
        # Calculate key insights
        insights = []
        
        # Peak hour analysis
        df['hour'] = pd.to_datetime(df['order_date']).dt.hour
        peak_hour = df.groupby('hour').size().idxmax()
        peak_orders = df.groupby('hour').size().max()
        insights.append(f"📈 Peak ordering hour: {peak_hour}:00 with {peak_orders} orders")
        
        # Best performing city
        city_revenue = df.groupby('city')['order_value'].sum()
        best_city = city_revenue.idxmax()
        best_city_revenue = city_revenue.max()
        insights.append(f"🏆 Top performing city: {best_city} (₹{best_city_revenue:,.0f} revenue)")
        
        # Average delivery time insights
        avg_delivery = df['delivery_time'].mean()
        if avg_delivery <= 30:
            delivery_status = "Excellent"
        elif avg_delivery <= 40:
            delivery_status = "Good"
        else:
            delivery_status = "Needs Improvement"
        insights.append(f"⏱️ Delivery performance: {delivery_status} ({avg_delivery:.1f} min average)")
        
        # Customer repeat rate
        customer_orders = df.groupby('customer_id').size()
        repeat_customers = (customer_orders > 1).sum()
        total_customers = len(customer_orders)
        repeat_rate = (repeat_customers / total_customers) * 100
        insights.append(f"🔄 Customer repeat rate: {repeat_rate:.1f}% ({repeat_customers}/{total_customers})")
        
        # Display insights
        for insight in insights:
            st.info(insight)
    
    def run_dashboard(self):
        """Main dashboard function"""
        st.title("🍔 Food Delivery Analytics Dashboard")
        st.markdown("Real-time insights into food delivery operations across multiple cities")
        
        # Sidebar filters
        st.sidebar.title("📊 Filters")
        
        # Load data
        with st.spinner("Loading data..."):
            df = self.load_data()
        
        if df.empty:
            st.error("No data available. Please check your database connection.")
            return
        
        # Date range filter
        min_date = pd.to_datetime(df['order_date']).min().date()
        max_date = pd.to_datetime(df['order_date']).max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - timedelta(days=30), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # City filter
        cities = ['All'] + sorted(df['city'].unique().tolist())
        selected_city = st.sidebar.selectbox("Select City", cities)
        
        # Apply filters
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                (pd.to_datetime(df['order_date']).dt.date >= start_date) &
                (pd.to_datetime(df['order_date']).dt.date <= end_date)
            ]
        
        if selected_city != 'All':
            df = df[df['city'] == selected_city]
        
        # Display metrics
        st.markdown("### 📈 Key Performance Indicators")
        self.create_metrics_cards(df)
        
        # Main charts
        st.markdown("### 📊 Performance Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend
            revenue_chart = self.create_revenue_trend_chart(df)
            st.plotly_chart(revenue_chart, use_container_width=True)
        
        with col2:
            # City performance
            city_chart = self.create_city_performance_chart(df)
            st.plotly_chart(city_chart, use_container_width=True)
        
        # Customer segmentation
        st.markdown("### 👥 Customer Segmentation Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            segmentation_chart, profiles = self.create_customer_segmentation_chart(df)
            if segmentation_chart:
                st.plotly_chart(segmentation_chart, use_container_width=True)
        
        with col2:
            if profiles:
                st.markdown("#### Segment Characteristics")
                for segment, profile in profiles.items():
                    with st.expander(f"{profile['characteristics']} ({profile['size']} customers)"):
                        st.write(f"**Average Spending:** ₹{profile['avg_total_spent']:.0f}")
                        st.write(f"**Order Frequency:** {profile['avg_order_frequency']:.1f} orders")
                        st.write(f"**Average Order Value:** ₹{profile['avg_order_value']:.0f}")
                        st.write(f"**Average Delivery Time:** {profile['avg_delivery_time']:.1f} minutes")
        
        # Operational analytics
        st.markdown("### 🚚 Operational Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Delivery performance
            delivery_chart = self.create_delivery_performance_chart(df)
            st.plotly_chart(delivery_chart, use_container_width=True)
        
        with col2:
            # Hourly heatmap
            heatmap_chart = self.create_hourly_orders_heatmap(df)
            st.plotly_chart(heatmap_chart, use_container_width=True)
        
        # Insights section
        self.display_operational_insights(df)
        
        # Data export section
        st.markdown("### 📥 Data Export")
        if st.button("Download Current Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"food_delivery_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        # Footer
        st.markdown("---")
        st.markdown(
            f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"**Total Records:** {len(df):,} | "
            f"**Date Range:** {df['order_date'].min()} to {df['order_date'].max()}"
        )

if __name__ == "__main__":
    dashboard = FoodDeliveryDashboard()
    dashboard.run_dashboard()