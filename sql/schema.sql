-- UTC Unix milliseconds throughout. Item/catalog relations are logical rather
-- than foreign keys because source metadata coverage is incomplete.
CREATE TABLE events (
 event_id INTEGER PRIMARY KEY,
 timestamp INTEGER NOT NULL,
 visitorid INTEGER NOT NULL,
 event TEXT NOT NULL CHECK(event IN ('view','addtocart','transaction')),
 itemid INTEGER NOT NULL,
 transactionid INTEGER
);
CREATE TABLE item_properties (
 timestamp INTEGER NOT NULL,
 itemid INTEGER NOT NULL,
 property TEXT NOT NULL,
 value TEXT NOT NULL,
 source_file TEXT NOT NULL
);
CREATE TABLE categories (
 categoryid INTEGER PRIMARY KEY,
 parentid INTEGER
);
CREATE TABLE category_history (
 itemid INTEGER NOT NULL,
 timestamp INTEGER NOT NULL,
 categoryid INTEGER NOT NULL,
 PRIMARY KEY(itemid,timestamp)
);
CREATE TABLE event_categories (
 event_id INTEGER PRIMARY KEY REFERENCES events(event_id),
 timestamp INTEGER NOT NULL,
 visitorid INTEGER NOT NULL,
 event TEXT NOT NULL,
 itemid INTEGER NOT NULL,
 categoryid INTEGER,
 category_timestamp INTEGER
);
