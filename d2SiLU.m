% derivata functiei dSiLU
function dg = d2SiLU(z)
    sigmoid = 1 ./ (1 + exp(-z)); 
    dg = sigmoid .* (1 - sigmoid) + sigmoid .* (1 - sigmoid) .* (1 - 2 * sigmoid) + z .* sigmoid .* (1 - sigmoid) .* (1 - 2 * sigmoid);
end