-- Core shop schema.

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    total_cents INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- Recompute an order total.
CREATE OR REPLACE FUNCTION order_total(oid INTEGER) RETURNS INTEGER AS $$
  SELECT sum(price * qty) FROM order_items WHERE order_id = oid;
$$ LANGUAGE sql;

CREATE INDEX idx_orders_created ON orders (created_at);
