% functia de activare (20)
function g = dSiLU(z)
    sigmoid = 1 ./ (1 + exp(-z));
    g = sigmoid .* (1 + z .* (1 - sigmoid));
end