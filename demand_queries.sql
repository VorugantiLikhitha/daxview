-- ═══════════════════════════════════════════════════════════════════════════
-- DaxView SQL Queries
-- Database: data/processed/demand.db  |  Table: demand
-- Columns: Product_Code, Warehouse, Product_Category, Date, Order_Demand
-- ═══════════════════════════════════════════════════════════════════════════


-- 1. Total demand by product category (ranked)
SELECT
    Product_Category,
    SUM(Order_Demand)                         AS total_demand,
    COUNT(DISTINCT Product_Code)              AS unique_skus,
    ROUND(AVG(Order_Demand), 0)               AS avg_order_size
FROM demand
GROUP BY Product_Category
ORDER BY total_demand DESC;


-- 2. Monthly demand trend (time series)
SELECT
    STRFTIME('%Y-%m', Date)                   AS month,
    SUM(Order_Demand)                         AS monthly_demand,
    COUNT(DISTINCT Product_Code)              AS active_skus
FROM demand
GROUP BY month
ORDER BY month;


-- 3. Warehouse performance ranking
SELECT
    Warehouse,
    SUM(Order_Demand)                         AS total_demand,
    COUNT(DISTINCT Product_Category)          AS categories_served,
    ROUND(AVG(Order_Demand), 0)               AS avg_order_demand
FROM demand
GROUP BY Warehouse
ORDER BY total_demand DESC;


-- 4. Seasonality: average demand by calendar month
SELECT
    CAST(STRFTIME('%m', Date) AS INTEGER)     AS month_num,
    CASE CAST(STRFTIME('%m', Date) AS INTEGER)
        WHEN 1 THEN 'Jan' WHEN 2 THEN 'Feb' WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr' WHEN 5 THEN 'May' WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul' WHEN 8 THEN 'Aug' WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct' WHEN 11 THEN 'Nov' WHEN 12 THEN 'Dec'
    END                                       AS month_name,
    ROUND(AVG(Order_Demand), 0)               AS avg_demand
FROM demand
GROUP BY month_num
ORDER BY month_num;


-- 5. Top 10 highest-demand SKUs
SELECT
    Product_Code,
    Product_Category,
    SUM(Order_Demand)                         AS total_demand,
    COUNT(*)                                  AS order_count,
    ROUND(AVG(Order_Demand), 0)               AS avg_per_order
FROM demand
GROUP BY Product_Code, Product_Category
ORDER BY total_demand DESC
LIMIT 10;


-- 6. Year-over-year demand comparison
SELECT
    STRFTIME('%Y', Date)                      AS year,
    Product_Category,
    SUM(Order_Demand)                         AS annual_demand
FROM demand
GROUP BY year, Product_Category
ORDER BY Product_Category, year;


-- 7. Stockout risk: categories with highest demand variance (CV)
SELECT
    Product_Category,
    ROUND(AVG(Order_Demand), 0)               AS mean_demand,
    ROUND(
        SQRT(AVG(Order_Demand * Order_Demand) - AVG(Order_Demand) * AVG(Order_Demand)),
        0
    )                                         AS std_demand,
    ROUND(
        SQRT(AVG(Order_Demand * Order_Demand) - AVG(Order_Demand) * AVG(Order_Demand))
        / NULLIF(AVG(Order_Demand), 0) * 100,
        1
    )                                         AS cv_pct
FROM demand
GROUP BY Product_Category
HAVING COUNT(*) > 20
ORDER BY cv_pct DESC
LIMIT 10;
