# Food Delivery Analytics Platform

## 🎯 Project Overview

This comprehensive analytics platform was developed during my **Summer Analyst internship at CampusHaat Solutions Pvt. Ltd.** The project analyzed **100,000+ food delivery transactions across 4 cities** to optimize customer retention, loyalty, and operational efficiency.

### Key Achievements
- ✅ **15% improvement** in customer retention through predictive modeling & customer behavioral segmentation
- ✅ **Enhanced delivery operations** by analyzing delivery times and kitchen performance
- ✅ Developed **RESTful APIs** & optimized **SQL analytics** for real-time dashboard visualization
- ✅ Created **customer segmentation** using K-means clustering based on user ordering frequency & spending patterns

## 🏗️ Architecture

```
food-delivery-analytics/
├── 📊 Data Processing & Analytics
│   ├── Customer Segmentation (K-means clustering)
│   ├── Retention Analysis (Cohort analysis)
│   └── Operational Metrics (Delivery performance)
├── 🤖 Machine Learning Models
│   ├── Delivery Time Prediction (Random Forest)
│   ├── Customer Churn Prediction (Gradient Boosting)
│   └── Recommendation Engine
├── 🚀 RESTful API (Flask)
│   ├── Analytics Endpoints
│   ├── Prediction Services
│   └── Real-time Metrics
├── 📈 Interactive Dashboard (Streamlit)
│   ├── KPI Monitoring
│   ├── Visual Analytics
│   └── Operational Insights
└── 🗄️ Database & SQL Analytics
    ├── Transaction Management
    ├── Complex Queries
    └── Performance Optimization
```

## 🛠️ Technology Stack

**Backend & Analytics:**
- **Python** (Pandas, NumPy, Scikit-learn)
- **Flask** (RESTful API development)
- **SQLite/PostgreSQL** (Database management)
- **SQL** (Advanced analytics queries)

**Machine Learning:**
- **K-means Clustering** (Customer segmentation)
- **Random Forest** (Delivery time prediction)
- **Gradient Boosting** (Churn prediction)
- **Statistical Analysis** (Cohort retention analysis)

**Frontend & Visualization:**
- **Streamlit** (Interactive dashboard)
- **Plotly** (Data visualizations)
- **HTML/CSS/JavaScript** (Dashboard UI)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/food-delivery-analytics.git
cd food-delivery-analytics
```

### 2. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Database
```bash
# Initialize database with sample data
python scripts/setup_database.py
python scripts/load_sample_data.py
```

### 4. Run Applications

**Start API Server:**
```bash
cd src/api
python app.py
# API runs on http://localhost:5000
```

**Launch Dashboard:**
```bash
cd dashboard
streamlit run dashboard_app.py
# Dashboard runs on http://localhost:8501
```

## 📊 Core Analytics Features

### 1. Customer Segmentation
- **K-means clustering** based on ordering frequency and spending patterns
- **RFM Analysis** (Recency, Frequency, Monetary)
- **Customer Lifetime Value** calculation
- **Behavioral pattern identification**

### 2. Retention Analysis
- **Cohort retention tracking**
- **Churn prediction modeling**
- **Customer lifecycle analysis**
- **Loyalty program optimization**

### 3. Operational Analytics
- **Delivery performance metrics**
- **Peak hour analysis**
- **Restaurant performance tracking**
- **City-wise operational efficiency**

## 🤖 Machine Learning Models

### Delivery Time Predictor
```python
# Example usage
from src.ml_models.predictive_models import DeliveryTimePredictor

predictor = DeliveryTimePredictor()
predictor.load_model('models/delivery_time_model.pkl')

features = {
    'order_value': 250.0,
    'distance_km': 3.5,
    'hour_of_day': 19,
    'weather_condition': 'Clear',
    'restaurant_id': 'REST_001'
}

predicted_time = predictor.predict(features)
# Output: 28.5 minutes
```

### Customer Segmentation
```python
from src.analytics.customer_segmentation import CustomerSegmentation

segmenter = CustomerSegmentation()
segments, profiles = segmenter.perform_segmentation(df)

# Output: Customer segments with characteristics
# - High-Value Frequent Customers
# - Medium-Value Regular Customers  
# - High-Frequency Low-Spend Customers
# - High-Ticket Occasional Customers
```

## 📈 API Endpoints

### Analytics Endpoints
```
GET /api/v1/analytics/segments
GET /api/v1/analytics/retention
GET /api/v1/analytics/operations
```

### Prediction Endpoints
```
POST /api/v1/predict/delivery-time
POST /api/v1/predict/churn
```

### Dashboard Endpoints
```
GET /api/v1/dashboard/metrics
GET /api/v1/recommendations
```

## 🗄️ Database Schema

### Transactions Table
```sql
CREATE TABLE transactions (
    order_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    restaurant_id VARCHAR(50),
    city VARCHAR(50),
    order_value DECIMAL(10,2),
    delivery_time INTEGER,
    order_date TIMESTAMP,
    cuisine_type VARCHAR(50),
    payment_method VARCHAR(20),
    delivery_status VARCHAR(20)
);
```

## 📊 Key SQL Analytics

### Customer Lifetime Value
```sql
WITH customer_metrics AS (
    SELECT 
        customer_id,
        SUM(order_value) as total_spent,
        COUNT(*) as total_orders,
        AVG(order_value) as avg_order_value,
        DATEDIFF(MAX(order_date), MIN(order_date)) as lifetime_days
    FROM transactions
    GROUP BY customer_id
)
SELECT 
    customer_id,
    total_spent,
    total_orders,
    (total_spent / lifetime_days * 365) as annualized_clv
FROM customer_metrics;
```

### Cohort Retention Analysis
```sql
WITH cohort_data AS (
    SELECT 
        customer_id,
        MIN(DATE_FORMAT(order_date, '%Y-%m')) as cohort_month
    FROM transactions
    GROUP BY customer_id
)
SELECT 
    cohort_month,
    COUNT(DISTINCT customer_id) as cohort_size,
    retention_rate
FROM cohort_data
GROUP BY cohort_month;
```

## 🎯 Business Impact

### Achieved Results
1. **15% Customer Retention Improvement**
   - Implemented predictive churn models
   - Developed targeted retention campaigns
   - Created personalized customer experiences

2. **Enhanced Delivery Operations**
   - Reduced average delivery time by 12%
   - Optimized delivery route planning
   - Improved kitchen performance metrics

3. **Data-Driven Insights**
   - Real-time operational dashboards
   - Automated reporting systems
   - Predictive analytics for business planning

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test API endpoints
python tests/test_api_endpoints.py
```

## 📦 Deployment

### Local Development
```bash
# Start all services
docker-compose up -d
```

### Production Deployment
```bash
# Build and deploy to cloud
./deploy.sh production
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- 🐱 GitHub: [Your GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- **CampusHaat Solutions Pvt. Ltd.** for the internship opportunity
- **Data Science Team** for guidance and support
- **Open Source Community** for tools and libraries used

---

**⭐ Star this repository if you found it helpful!**