SELECT 'daily' AS grain,date(timestamp/1000,'unixepoch') AS period,
COUNT(*) AS events, SUM(event='view') AS views,
SUM(event='addtocart') AS carts, SUM(event='transaction') AS transaction_events,
COUNT(DISTINCT visitorid) AS visitors,
1.0*COUNT(DISTINCT CASE WHEN event='transaction' THEN visitorid END)
/COUNT(DISTINCT visitorid) AS visitor_conversion_rate
FROM events GROUP BY period
UNION ALL
-- Monday week start; handles Sundays and Mondays without week-number ambiguity.
SELECT 'weekly',date(timestamp/1000,'unixepoch',
 '-' || ((CAST(strftime('%w',timestamp/1000,'unixepoch') AS INTEGER)+6)%7) || ' days'),
COUNT(*),SUM(event='view'),SUM(event='addtocart'),SUM(event='transaction'),
COUNT(DISTINCT visitorid),
1.0*COUNT(DISTINCT CASE WHEN event='transaction' THEN visitorid END)
/COUNT(DISTINCT visitorid)
FROM events GROUP BY 2;
