module Solver

# Newton iteration for square roots.
function newton_sqrt(x::Float64, iters::Int=10)
    g = x / 2
    for _ in 1:iters
        g = (g + x / g) / 2
    end
    g
end

mutable struct State
    guess::Float64
end

end
