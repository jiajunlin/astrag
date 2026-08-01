require 'json'

# Order management for the shop.
module Shop
  # Applies discounts and totals an order.
  class Order
    def initialize(items)
      @items = items
    end

    # Total price in cents, tax included.
    def total_with_tax(rate)
      subtotal = @items.sum { |i| i[:price] * i[:qty] }
      if subtotal > 10_000
        subtotal -= discount(subtotal)
      end
      (subtotal * (1 + rate)).round
    end

    def discount(subtotal)
      (subtotal * 0.05).round
    end
  end
end
