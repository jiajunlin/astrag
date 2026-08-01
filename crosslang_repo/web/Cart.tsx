/** Cart panel: loads the current cart and submits checkout. */
export async function loadCart(): Promise<CartResponse> {
  const res = await fetch('/api/cart');
  return res.json();
}

/** Submit the order to the Rust order service. */
export async function submitOrder(orderId: string, payload: Order): Promise<void> {
  await axios.post(`/api/orders/${orderId}`, payload);
}

interface CartResponse { items: unknown[]; totalCents: number; }
interface Order { items: unknown[]; }
