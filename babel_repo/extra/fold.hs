-- Folding helpers.

-- Sum of squares of a list.
sumSquares :: [Int] -> Int
sumSquares xs = foldr (\x acc -> x * x + acc) 0 xs

mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)
