defmodule Shop.Pricing do
  @moduledoc "Price calculations."

  # Apply a percentage discount to cents.
  def discounted(cents, pct) do
    cents - div(cents * pct, 100)
  end

  defp clamp(x, lo, hi) do
    x |> max(lo) |> min(hi)
  end
end
