import React from 'react';
import { formatPrice } from '../app.js';

interface CartItem {
  sku: string;
  title: string;
  price: number;
  qty: number;
}

interface CartProps {
  items: CartItem[];
  onRemove: (sku: string) => void;
}

/** One row of the cart table. */
const CartRow = ({ item, onRemove }: { item: CartItem; onRemove: (s: string) => void }) => (
  <li className="cart-item">
    <span>{item.title}</span>
    <span>{formatPrice(item.price * item.qty)}</span>
    <button onClick={() => onRemove(item.sku)}>x</button>
  </li>
);

/** Shopping-cart panel: rows plus a grand total. */
export default function Cart({ items, onRemove }: CartProps) {
  const total = items.reduce((sum, i) => sum + i.price * i.qty, 0);
  return (
    <section id="cart-panel">
      <ul>
        {items.map((item) => (
          <CartRow key={item.sku} item={item} onRemove={onRemove} />
        ))}
      </ul>
      <footer>Total: {formatPrice(total)}</footer>
    </section>
  );
}
