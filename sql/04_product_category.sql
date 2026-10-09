-- Category observed at event time; unknown categories are retained.
SELECT 'category' AS grain, COALESCE(categoryid,'unknown') AS entity,
SUM(event='view') AS views, SUM(event='addtocart') AS carts,
SUM(event='transaction') AS transaction_events,
COUNT(DISTINCT visitorid) AS visitors,
1.0*COUNT(DISTINCT CASE WHEN event='transaction' THEN visitorid END)
/COUNT(DISTINCT visitorid) AS visitor_conversion_rate
FROM event_categories GROUP BY categoryid
UNION ALL
SELECT 'product', CAST(itemid AS TEXT), SUM(event='view'),SUM(event='addtocart'),
SUM(event='transaction'),COUNT(DISTINCT visitorid),
1.0*COUNT(DISTINCT CASE WHEN event='transaction' THEN visitorid END)
/COUNT(DISTINCT visitorid)
FROM events GROUP BY itemid;
