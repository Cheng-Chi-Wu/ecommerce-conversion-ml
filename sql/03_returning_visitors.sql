-- Returning means observed on at least two different UTC dates.
WITH visitors AS (
 SELECT visitorid, COUNT(DISTINCT date(timestamp/1000,'unixepoch')) AS active_days,
 SUM(event='transaction') AS transaction_events,
 COUNT(*) AS events FROM events GROUP BY visitorid
)
SELECT CASE WHEN active_days>1 THEN 'returning' ELSE 'single_day' END AS segment,
COUNT(*) AS visitors, AVG(events) AS mean_events,
AVG(transaction_events>0) AS visitor_conversion_rate
FROM visitors GROUP BY segment;
