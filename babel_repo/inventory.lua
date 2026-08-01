-- Inventory helpers for the game store.

-- Restock an item up to its cap.
function restock(item, amount)
  local cap = item.cap or 100
  item.count = math.min(item.count + amount, cap)
  return item.count
end

local function audit(items)
  for _, it in ipairs(items) do
    print(it.name, it.count)
  end
end

M = {}
M.value = function(item)
  return item.count * item.price
end
