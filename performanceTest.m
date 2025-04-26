% se calculeaza indicatorii de performanta pentru problema de regresie
function [] = performanceTest(e, grad, SGD)
    fprintf("Indicatorii de performanta (GRADIENT vs SGD):\n");
    ma = mean(e);
    m = sum((e - ma) .^ 2);

    % calcul indicatori pentru metoda gradient
    R2   = 1 - (sum((e - grad) .^ 2) ./ m);
    MAE = mean(abs(e - grad));
    MSE = mean((e - grad) .^ 2);
    fprintf("R^2: %f | MAE: %f | MSE: %f\n", R2, MAE, MSE);

    % calcul indicatori pentru metoda gradient stocastica
    R2   = 1 - (sum((e - SGD) .^ 2) ./ m);
    MAE = mean(abs(e - SGD));
    MSE = mean((e - SGD) .^ 2);
    fprintf("R^2: %f | MAE: %f | MSE: %f\n", R2, MAE, MSE);
end