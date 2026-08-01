using System;
using System.Collections.Generic;
using System.Linq;

namespace Shop;

/// <summary>Line item in an order.</summary>
public class Item
{
    public string Sku = "";
    public decimal Price;
    public int Quantity;
}

/// <summary>Computes order totals, discounts and tax.</summary>
public class OrderService
{
    private const decimal TaxRate = 0.0825m;

    /// <summary>Total for all items including tax, in dollars.</summary>
    public decimal Total(IEnumerable<Item> items)
    {
        var subtotal = items.Sum(i => i.Price * i.Quantity);
        return ApplyTax(subtotal);
    }

    /// <summary>Apply the configured sales-tax rate to a subtotal.</summary>
    private static decimal ApplyTax(decimal subtotal)
    {
        return Math.Round(subtotal * (1 + TaxRate), 2);
    }
}
