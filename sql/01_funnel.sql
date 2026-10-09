-- Ordered visitor funnel over the full capture period; strict timestamps.
-- This describes observed visitors, not a same-session attribution model.
WITH views AS (
 SELECT visitorid, MIN(timestamp) AS viewed_at FROM events
 WHERE event='view' GROUP BY visitorid
), carts AS (
 SELECT v.visitorid, v.viewed_at, MIN(e.timestamp) AS cart_at
 FROM views v JOIN events e ON e.visitorid=v.visitorid
 AND e.timestamp>v.viewed_at AND e.event='addtocart'
 GROUP BY v.visitorid
), purchases AS (
 SELECT c.visitorid, MIN(e.timestamp) AS purchased_at
 FROM carts c JOIN events e ON e.visitorid=c.visitorid
 AND e.timestamp>c.cart_at AND e.event='transaction'
 GROUP BY c.visitorid
)
SELECT 'view' AS stage, COUNT(*) AS visitors FROM views
UNION ALL SELECT 'addtocart', COUNT(*) FROM carts
UNION ALL SELECT 'transaction', COUNT(*) FROM purchases;
