% gradientul functiei de pierdere
function [LX, Lx] = dMSE(e, A, X, x)
    Z = A * X; Z1 = dSiLU(Z);
    error = Z1 * x  - e; N = size(e, 1);

    LX = (A' * ((error * x') .* d2SiLU(Z))) / N;    % gradient dupa X
    Lx = Z1' * error / N;                           % gradient dupa x
end