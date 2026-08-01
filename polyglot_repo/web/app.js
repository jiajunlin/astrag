import { CART_KEY } from './config.js';

/** Format a price in cents as a localized currency string. */
export function formatPrice(cents, currency = 'USD') {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(cents / 100);
}

/** Debounce `fn`, firing only after `ms` of quiet. */
export const debounce = (fn, ms = 200) => {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
};

/** In-memory cart with derived totals. */
class CartStore {
  constructor() {
    this.items = [];
  }

  add(item) {
    this.items.push(item);
    console.log(`added ${item.sku} (${this.items.length} items)`);
  }

  totalCents() {
    return this.items.reduce((sum, i) => sum + i.price * i.qty, 0);
  }
}

export const cart = new CartStore();
