#standardSQL
	#duplicates
SELECT
  COUNT(*) AS num_duplicate_rows,
  *
FROM
  `data-to-insights.ecommerce.all_sessions_raw`
GROUP BY
  fullVisitorId,
  channelGrouping,
  time,
  country,
  city,
  totalTransactionRevenue,
  transactions,
  timeOnSite,
  pageviews,
  sessionQualityDim,
  date,
  visitId,
  type,
  productRefundAmount,
  productQuantity,
  productPrice,
  productRevenue,
  productSKU,
  v2ProductName,
  v2ProductCategory,
  productVariant,
  currencyCode,
  itemQuantity,
  itemRevenue,
  transactionRevenue,
  transactionId,
  pageTitle,
  searchKeyword,
  pagePathLevel1,
  eCommerceAction_type,
  eCommerceAction_step,
  eCommerceAction_option
HAVING
  num_duplicate_rows > 1;
  
#standardSQL
	#how many production with distinct visitors
SELECT
  COUNT(*) AS product_views,
  COUNT(DISTINCT fullVisitorId) AS unique_visitors
FROM
  `data-to-insights.ecommerce.all_sessions`;
  
#standardSQL
	#deduplicates GROUP BY
SELECT
  (v2ProductName) AS ProductName
FROM `data-to-insights.ecommerce.all_sessions`
GROUP BY ProductName
ORDER BY ProductName

WITH
  unique_product_views_by_person AS (
    -- find each unique product viewed by each visitor
  SELECT
    fullVisitorId,
    (v2ProductName) AS ProductName
  FROM
    `data-to-insights.ecommerce.all_sessions`
  WHERE
    type = 'PAGE'
  GROUP BY
    fullVisitorId,
    v2ProductName )
  -- aggregate the top viewed products and sort them
SELECT
  COUNT(*) AS unique_view_count,
  ProductName
FROM
  unique_product_views_by_person
GROUP BY
  ProductName
ORDER BY
  unique_view_count DESC
LIMIT
  5
  
#standardSQL
	#More than 1000 units were added to a cart or ordered
    #AND are not frisbees

SELECT
  COUNT(*) AS product_views,
  COUNT(productQuantity) AS potential_orders,
  SUM(productQuantity) AS quantity_product_added,
  (COUNT(productQuantity) / COUNT(*)) AS conversion_rate,
  v2ProductName
FROM
  `data-to-insights.ecommerce.all_sessions`
WHERE
  LOWER(v2ProductName) NOT LIKE '%frisbee%'
GROUP BY
  v2ProductName
HAVING
  quantity_product_added > 1000
ORDER BY
  conversion_rate DESC
LIMIT
  10;
  
'''
#standardSQL
SELECT 
  COUNT(*) AS product_views,
  COUNT(productQuantity) AS potential_orders,
  SUM(productQuantity) AS quantity_product_added,
  (COUNT(productQuantity) / COUNT(*)) AS conversion_rate,
  v2ProductName
FROM `data-to-insights.ecommerce.all_sessions`
WHERE LOWER(v2ProductName) NOT LIKE '%frisbee%' 
GROUP BY v2ProductName
HAVING quantity_product_added > 1000 
ORDER BY conversion_rate DESC
LIMIT 10;
'''

'''
#standardSQL
SELECT 
  COUNT(DISTINCT fullVisitorId) AS number_of_unique_visitors,
  eCommerceAction_type,
  CASE eCommerceAction_type
  WHEN '0' THEN 'Unknown'
  WHEN '1' THEN 'Click through of product lists'
  WHEN '2' THEN 'Product detail views'
  WHEN '3' THEN 'Add product(s) to cart'
  WHEN '4' THEN 'Remove product(s) from cart'
  WHEN '5' THEN 'Check out'
  WHEN '6' THEN 'Completed purchase'
  WHEN '7' THEN 'Refund of purchase'
  WHEN '8' THEN 'Checkout options'
  ELSE 'ERROR'
  END AS eCommerceAction_type_label
FROM `data-to-insights.ecommerce.all_sessions` 
GROUP BY eCommerceAction_type
ORDER BY eCommerceAction_type;
'''

'''
#standardSQL
# high quality abandoned carts
SELECT  
  #unique_session_id
  CONCAT(fullVisitorId,CAST(visitId AS STRING)) AS unique_session_id,
  sessionQualityDim,
  SUM(productRevenue) AS transaction_revenue,
  MAX(eCommerceAction_type) AS checkout_progress
FROM `data-to-insights.ecommerce.all_sessions` 
WHERE sessionQualityDim > 60 # high quality session
GROUP BY unique_session_id, sessionQualityDim
HAVING 
  checkout_progress = '3' # 3 = added to cart
  AND (transaction_revenue = 0 OR transaction_revenue IS NULL)
'''

#standardSQL
  # high quality abandoned carts
  # customer segmentation
SELECT
  #unique_session_id
  CONCAT(fullVisitorId,CAST(visitId AS STRING)) AS unique_session_id,
  sessionQualityDim,
  SUM(productRevenue) AS transaction_revenue,
  MAX(eCommerceAction_type) AS checkout_progress
FROM
  `data-to-insights.ecommerce.all_sessions`
WHERE
  sessionQualityDim > 60 # high quality session
GROUP BY
  unique_session_id,
  sessionQualityDim
HAVING
  checkout_progress = '3' # 3 = added to cart
  AND (transaction_revenue = 0
    OR transaction_revenue IS NULL)