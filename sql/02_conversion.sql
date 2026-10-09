-- Visitor purchase rate and cart abandonment with complete 24h follow-up.
WITH bounds AS (SELECT MAX(timestamp) AS end_at FROM events),
eligible_carts AS (
 SELECT e.* FROM events e,bounds b WHERE e.event='addtocart'
 AND e.timestamp+86400000<=b.end_at
), cart_outcomes AS (
 SELECT c.*, EXISTS(SELECT 1 FROM events t
 WHERE t.visitorid=c.visitorid AND t.itemid=c.itemid
 AND t.event='transaction' AND t.timestamp>c.timestamp
 AND t.timestamp<=c.timestamp+86400000) AS purchased
 FROM eligible_carts c
)
SELECT 'visitor_conversion_rate' AS metric,
 1.0*COUNT(DISTINCT CASE WHEN event='transaction' THEN visitorid END)
 /COUNT(DISTINCT visitorid) AS value FROM events
UNION ALL SELECT '24h_cart_event_abandonment_rate',1.0-AVG(purchased) FROM cart_outcomes
UNION ALL SELECT 'eligible_cart_events',COUNT(*) FROM cart_outcomes;
