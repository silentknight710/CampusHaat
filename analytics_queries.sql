-- sql_queries/analytics_queries.sql

-- ===============================================
-- CUSTOMER SEGMENTATION & RETENTION ANALYSIS
-- ===============================================

-- 1. Customer Lifetime Value (CLV) Calculation
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) as total_orders,
        SUM(order_value) as total_spent,
        AVG(order_value) as avg_order_value,
        MIN(order_date) as first_order_date,
        MAX(order_date) as last_order_date,
        DATEDIFF(MAX(order_date), MIN(order_date)) + 1 as customer_lifetime_days
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY customer_id
),
clv_calculation AS (
    SELECT 
        customer_id,
        total_orders,
        total_spent,
        avg_order_value,
        customer_lifetime_days,
        CASE 
            WHEN customer_lifetime_days > 0 
            THEN (total_orders / customer_lifetime_days) * 365.25 
            ELSE total_orders 
        END as annual_order_frequency,
        CASE 
            WHEN customer_lifetime_days > 0 
            THEN total_spent / customer_lifetime_days * 365.25 
            ELSE total_spent 
        END as annualized_clv
    FROM customer_metrics
)
SELECT 
    customer_id,
    total_spent,
    total_orders,
    avg_order_value,
    customer_lifetime_days,
    ROUND(annual_order_frequency, 2) as projected_annual_orders,
    ROUND(annualized_clv, 2) as customer_lifetime_value,
    CASE 
        WHEN annualized_clv >= 2000 THEN 'High Value'
        WHEN annualized_clv >= 1000 THEN 'Medium Value'
        WHEN annualized_clv >= 500 THEN 'Low Value'
        ELSE 'Very Low Value'
    END as customer_segment
FROM clv_calculation
ORDER BY annualized_clv DESC;

-- 2. Cohort Retention Analysis
WITH cohort_data AS (
    SELECT 
        customer_id,
        MIN(DATE_FORMAT(order_date, '%Y-%m')) as cohort_month,
        order_date
    FROM transactions
    GROUP BY customer_id, order_date
),
cohort_table AS (
    SELECT 
        cohort_month,
        customer_id,
        TIMESTAMPDIFF(MONTH, STR_TO_DATE(CONCAT(cohort_month, '-01'), '%Y-%m-%d'), order_date) as month_number
    FROM cohort_data cd
    JOIN transactions t ON cd.customer_id = t.customer_id
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM cohort_data
    GROUP BY cohort_month
),
retention_table AS (
    SELECT 
        ct.cohort_month,
        ct.month_number,
        COUNT(DISTINCT ct.customer_id) as customers_retained
    FROM cohort_table ct
    GROUP BY ct.cohort_month, ct.month_number
)
SELECT 
    rt.cohort_month,
    rt.month_number,
    rt.customers_retained,
    cs.cohort_size,
    ROUND(100.0 * rt.customers_retained / cs.cohort_size, 2) as retention_rate
FROM retention_table rt
JOIN cohort_sizes cs ON rt.cohort_month = cs.cohort_month
WHERE rt.month_number <= 12  -- First 12 months
ORDER BY rt.cohort_month, rt.month_number;

-- 3. RFM Analysis (Recency, Frequency, Monetary)
WITH rfm_data AS (
    SELECT 
        customer_id,
        DATEDIFF(CURDATE(), MAX(order_date)) as recency_days,
        COUNT(DISTINCT order_id) as frequency,
        SUM(order_value) as monetary_value
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT 
        customer_id,
        recency_days,
        frequency,
        monetary_value,
        NTILE(5) OVER (ORDER BY recency_days ASC) as recency_score,
        NTILE(5) OVER (ORDER BY frequency DESC) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) as monetary_score
    FROM rfm_data
)
SELECT 
    customer_id,
    recency_days,
    frequency,
    ROUND(monetary_value, 2) as monetary_value,
    recency_score,
    frequency_score,
    monetary_score,
    CONCAT(recency_score, frequency_score, monetary_score) as rfm_segment,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 THEN 'Potential Loyalists'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Cannot Lose Them'
        WHEN recency_score >= 3 AND frequency_score <= 1 THEN 'New Customers'
        ELSE 'Others'
    END as customer_category
FROM rfm_scores
ORDER BY monetary_score DESC, frequency_score DESC, recency_score DESC;

-- ===============================================
-- OPERATIONAL EFFICIENCY ANALYSIS
-- ===============================================

-- 4. Delivery Performance by City and Time Period
SELECT 
    city,
    DATE_FORMAT(order_date, '%Y-%m') as month,
    COUNT(DISTINCT order_id) as total_orders,
    AVG(delivery_time) as avg_delivery_time,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delivery_time) as median_delivery_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY delivery_time) as p95_delivery_time,
    COUNT(CASE WHEN delivery_time <= 30 THEN 1 END) / COUNT(*) * 100 as on_time_delivery_rate,
    SUM(order_value) as total_revenue,
    AVG(order_value) as avg_order_value
FROM transactions
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
GROUP BY city, DATE_FORMAT(order_date, '%Y-%m')
ORDER BY city, month DESC;

-- 5. Peak Hours Analysis
WITH hourly_orders AS (
    SELECT 
        HOUR(order_date) as order_hour,
        DAYNAME(order_date) as day_of_week,
        COUNT(*) as order_count,
        AVG(delivery_time) as avg_delivery_time,
        SUM(order_value) as hourly_revenue
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
    GROUP BY HOUR(order_date), DAYNAME(order_date)
),
peak_identification AS (
    SELECT 
        order_hour,
        day_of_week,
        order_count,
        avg_delivery_time,
        hourly_revenue,
        CASE 
            WHEN order_count >= (SELECT AVG(order_count) * 1.5 FROM hourly_orders) THEN 'Peak'
            WHEN order_count >= (SELECT AVG(order_count) FROM hourly_orders) THEN 'Normal'
            ELSE 'Low'
        END as demand_level
    FROM hourly_orders
)
SELECT 
    day_of_week,
    order_hour,
    order_count,
    ROUND(avg_delivery_time, 2) as avg_delivery_time,
    ROUND(hourly_revenue, 2) as hourly_revenue,
    demand_level,
    ROUND(order_count / SUM(order_count) OVER (PARTITION BY day_of_week) * 100, 2) as pct_of_daily_orders
FROM peak_identification
ORDER BY day_of_week, order_hour;

-- 6. Restaurant Performance Analysis
WITH restaurant_metrics AS (
    SELECT 
        restaurant_id,
        cuisine_type,
        city,
        COUNT(DISTINCT order_id) as total_orders,
        AVG(delivery_time) as avg_delivery_time,
        SUM(order_value) as total_revenue,
        AVG(order_value) as avg_order_value,
        COUNT(DISTINCT customer_id) as unique_customers,
        MIN(order_date) as first_order_date,
        MAX(order_date) as last_order_date
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    GROUP BY restaurant_id, cuisine_type, city
),
restaurant_rankings AS (
    SELECT 
        *,
        RANK() OVER (PARTITION BY city ORDER BY total_revenue DESC) as revenue_rank,
        RANK() OVER (PARTITION BY city ORDER BY total_orders DESC) as order_rank,
        RANK() OVER (PARTITION BY city ORDER BY avg_delivery_time ASC) as delivery_rank
    FROM restaurant_metrics
)
SELECT 
    restaurant_id,
    cuisine_type,
    city,
    total_orders,
    ROUND(avg_delivery_time, 2) as avg_delivery_time,
    ROUND(total_revenue, 2) as total_revenue,
    ROUND(avg_order_value, 2) as avg_order_value,
    unique_customers,
    ROUND(total_orders / unique_customers, 2) as avg_orders_per_customer,
    revenue_rank,
    order_rank,
    delivery_rank,
    CASE 
        WHEN revenue_rank <= 5 AND delivery_rank <= 10 THEN 'Top Performer'
        WHEN revenue_rank <= 10 THEN 'High Revenue'
        WHEN delivery_rank <= 5 THEN 'Fast Delivery'
        WHEN total_orders >= 100 THEN 'High Volume'
        ELSE 'Standard'
    END as performance_category
FROM restaurant_rankings
ORDER BY city, total_revenue DESC;

-- ===============================================
-- PREDICTIVE INSIGHTS & RECOMMENDATIONS
-- ===============================================

-- 7. Churn Risk Identification
WITH customer_activity AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) as total_orders,
        MAX(order_date) as last_order_date,
        DATEDIFF(CURDATE(), MAX(order_date)) as days_since_last_order,
        AVG(DATEDIFF(
            order_date, 
            LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date)
        )) as avg_days_between_orders,
        SUM(order_value) as total_spent,
        AVG(order_value) as avg_order_value
    FROM transactions
    GROUP BY customer_id
    HAVING total_orders >= 2  -- Only customers with multiple orders
),
churn_risk AS (
    SELECT 
        customer_id,
        total_orders,
        last_order_date,
        days_since_last_order,
        ROUND(avg_days_between_orders, 2) as avg_days_between_orders,
        ROUND(total_spent, 2) as total_spent,
        ROUND(avg_order_value, 2) as avg_order_value,
        CASE 
            WHEN days_since_last_order > (avg_days_between_orders * 3) THEN 'High Risk'
            WHEN days_since_last_order > (avg_days_between_orders * 2) THEN 'Medium Risk'
            WHEN days_since_last_order > avg_days_between_orders THEN 'Low Risk'
            ELSE 'Active'
        END as churn_risk_level
    FROM customer_activity
    WHERE avg_days_between_orders IS NOT NULL
)
SELECT 
    customer_id,
    total_orders,
    last_order_date,
    days_since_last_order,
    avg_days_between_orders,
    total_spent,
    avg_order_value,
    churn_risk_level,
    CASE 
        WHEN churn_risk_level = 'High Risk' AND total_spent >= 1000 THEN 'Priority Retention'
        WHEN churn_risk_level = 'High Risk' THEN 'Standard Retention'
        WHEN churn_risk_level = 'Medium Risk' AND total_orders >= 10 THEN 'Engagement Campaign'
        ELSE 'Monitor'
    END as recommended_action
FROM churn_risk
ORDER BY 
    CASE churn_risk_level 
        WHEN 'High Risk' THEN 1 
        WHEN 'Medium Risk' THEN 2 
        WHEN 'Low Risk' THEN 3 
        ELSE 4 
    END,
    total_spent DESC;

-- 8. Cross-sell Opportunities Analysis
WITH customer_cuisine_preferences AS (
    SELECT 
        customer_id,
        cuisine_type,
        COUNT(*) as orders_count,
        SUM(order_value) as spent_on_cuisine,
        RANK() OVER (PARTITION BY customer_id ORDER BY COUNT(*) DESC) as preference_rank
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
    GROUP BY customer_id, cuisine_type
),
primary_preferences AS (
    SELECT 
        customer_id,
        cuisine_type as primary_cuisine,
        orders_count as primary_orders,
        spent_on_cuisine as primary_spent
    FROM customer_cuisine_preferences
    WHERE preference_rank = 1
),
cuisine_combinations AS (
    SELECT 
        t1.cuisine_type as cuisine_1,
        t2.cuisine_type as cuisine_2,
        COUNT(DISTINCT t1.customer_id) as customers_ordering_both,
        AVG(t1.order_value + t2.order_value) as avg_combined_order_value
    FROM transactions t1
    JOIN transactions t2 ON t1.customer_id = t2.customer_id 
        AND t1.cuisine_type != t2.cuisine_type
        AND ABS(DATEDIFF(t1.order_date, t2.order_date)) <= 30
    GROUP BY t1.cuisine_type, t2.cuisine_type
    HAVING customers_ordering_both >= 10
)
SELECT 
    pp.customer_id,
    pp.primary_cuisine,
    pp.primary_orders,
    pp.primary_spent,
    cc.cuisine_2 as recommended_cuisine,
    cc.customers_ordering_both as market_validation,
    ROUND(cc.avg_combined_order_value, 2) as potential_order_value
FROM primary_preferences pp
JOIN cuisine_combinations cc ON pp.primary_cuisine = cc.cuisine_1
WHERE NOT EXISTS (
    SELECT 1 FROM customer_cuisine_preferences ccp
    WHERE ccp.customer_id = pp.customer_id 
    AND ccp.cuisine_type = cc.cuisine_2
)
ORDER BY pp.primary_spent DESC, cc.customers_ordering_both DESC;

-- ===============================================
-- BUSINESS OPTIMIZATION INSIGHTS
-- ===============================================

-- 9. City-wise Growth Opportunities
WITH monthly_city_metrics AS (
    SELECT 
        city,
        DATE_FORMAT(order_date, '%Y-%m') as month,
        COUNT(DISTINCT order_id) as orders,
        COUNT(DISTINCT customer_id) as customers,
        SUM(order_value) as revenue,
        AVG(delivery_time) as avg_delivery_time
    FROM transactions
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
    GROUP BY city, DATE_FORMAT(order_date, '%Y-%m')
),
city_growth AS (
    SELECT 
        city,
        month,
        orders,
        customers,
        revenue,
        avg_delivery_time,
        LAG(orders, 1) OVER (PARTITION BY city ORDER BY month) as prev_month_orders,
        LAG(revenue, 1) OVER (PARTITION BY city ORDER BY month) as prev_month_revenue
    FROM monthly_city_metrics
),
growth_calculations AS (
    SELECT 
        city,
        month,
        orders,
        customers,
        revenue,
        ROUND(avg_delivery_time, 2) as avg_delivery_time,
        CASE 
            WHEN prev_month_orders > 0 
            THEN ROUND((orders - prev_month_orders) / prev_month_orders * 100, 2)
            ELSE NULL 
        END as orders_growth_rate,
        CASE 
            WHEN prev_month_revenue > 0 
            THEN ROUND((revenue - prev_month_revenue) / prev_month_revenue * 100, 2)
            ELSE NULL 
        END as revenue_growth_rate
    FROM city_growth
)
SELECT 
    city,
    COUNT(*) as months_of_data,
    SUM(orders) as total_orders,
    SUM(customers) as total_customers,
    SUM(revenue) as total_revenue,
    AVG(avg_delivery_time) as overall_avg_delivery_time,
    AVG(orders_growth_rate) as avg_monthly_orders_growth,
    AVG(revenue_growth_rate) as avg_monthly_revenue_growth,
    CASE 
        WHEN AVG(orders_growth_rate) >= 10 THEN 'High Growth'
        WHEN AVG(orders_growth_rate) >= 5 THEN 'Moderate Growth'
        WHEN AVG(orders_growth_rate) >= 0 THEN 'Stable'
        ELSE 'Declining'
    END as growth_category,
    CASE 
        WHEN AVG(avg_delivery_time) <= 25 THEN 'Excellent'
        WHEN AVG(avg_delivery_time) <= 35 THEN 'Good'
        WHEN AVG(avg_delivery_time) <= 45 THEN 'Average'
        ELSE 'Needs Improvement'
    END as delivery_performance
FROM growth_calculations
WHERE orders_growth_rate IS NOT NULL
GROUP BY city
ORDER BY avg_monthly_revenue_growth DESC;

-- 10. Executive Summary Dashboard Query
SELECT 
    'Overall Performance' as metric_category,
    JSON_OBJECT(
        'total_orders_last_30_days', (
            SELECT COUNT(*) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        ),
        'total_revenue_last_30_days', (
            SELECT ROUND(SUM(order_value), 2) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        ),
        'avg_order_value_last_30_days', (
            SELECT ROUND(AVG(order_value), 2) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        ),
        'avg_delivery_time_last_30_days', (
            SELECT ROUND(AVG(delivery_time), 2) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        ),
        'active_customers_last_30_days', (
            SELECT COUNT(DISTINCT customer_id) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        ),
        'active_restaurants_last_30_days', (
            SELECT COUNT(DISTINCT restaurant_id) FROM transactions 
            WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        )
    ) as metrics;