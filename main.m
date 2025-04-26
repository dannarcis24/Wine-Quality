close all; clc; clear;

[A_learn, A_test, e_learn, e_test] = dataFromFile(1000);
[N, n] = size(A_learn);   % numarul de seturi pentru test && numarul de informatii

m = 120; % numarul de neuroni din stratul ascuns

X = randn(n + 1, m) * 0.01; x = randn(m, 1) * 0.01;   % parametrii optimi
A = [A_learn ones(N, 1)];

y = @(A, X, x) dSiLU(A * X) * x;        % iesirea
MSE = @(e, y) mean((e - y) .^ 2) / 2;   % functia de pierdere

% METODA GRADIENT
max_iter = 1000;  % initializari pentru implementarea metodei
pas = 0.01;
X1 = X; x1 = x;

grad_norm  = zeros(max_iter, 1); 
grad_error = zeros(max_iter, 1);
grad_time  = zeros(max_iter, 1);

tic;
for i=1:max_iter
    start = tic;
    [LX, Lx] = dMSE(e_learn, A, X1, x1);

    X1 = X1 - pas * LX;
    x1 = x1 - pas * Lx;

    % retinem informatiile
    grad_norm(i)  = norm(dMSE(e_learn, A, X1, x1));
    grad_error(i) = MSE(e_learn, y(A, X1, x1));
    grad_time(i)  = toc(start);
end
total = toc;
fprintf("Durata de executie totala a metodei gradient: %fs\n", total);

% METODA GRADIENT STOCASTIC (SGD)
max_iter = 1000;  % initializari pentru implementarea metodei
pas = 0.01;
X2 = X; x2 = x;

SGD_norm  = zeros(max_iter, 1);
SGD_error = zeros(max_iter, 1);
SGD_time  = zeros(max_iter, 1);

tic;
for i = 1:max_iter
    start = tic;
    
    idx = randi(N); % selectez un exemplu random
    Ai = [A_learn(idx, :) 1];

    % calculam valorile gradientilor
    Zi = Ai * X2;
    Z1i = dSiLU(Zi); 
    yi = Z1i * x2;

    err = yi - e_learn(idx);
    LX = Ai' * ((err * x2') .* d2SiLU(Zi));
    Lx = Z1i' * err;

    X2 = X2 - pas * LX;
    x2 = x2 - pas * Lx;

    % retinem informatiile
    SGD_norm(i)  = norm(dMSE(e_learn, A, X2, x2));
    SGD_error(i) = MSE(e_learn, y(A, X2, x2));
    SGD_time(i)  = toc(start);
end
total = toc;
fprintf("Durata de executie totala a metodei gradient stocastica: %fs\n\n", total);

% GRAFICE METODE
figure; grid on;    % norma gradient vs SGD
semilogy(1:max_iter, grad_norm); hold on; semilogy(1:max_iter, SGD_norm);
xlabel('Iteratii'); ylabel('Norma gradient'); title('Evolutia normei gradientului (Gradient vs SGD)');
legend('Gradient', 'SGD');

figure; grid on;    % fuctie pierdere gradient vs SGD
semilogy(1:max_iter, grad_error); hold on; semilogy(1:max_iter, SGD_error);
xlabel('Iteratii'); ylabel('Eroare'); title('Evolutia functiei obiectiv (Gradient vs SGD)');
legend('Gradient', 'SGD');

figure; grid on;    % timp gradient vs SGD
plot(1:max_iter, grad_time); hold on; plot(1:max_iter, SGD_time);
xlabel('Iteratii'); ylabel('Durata'); title('Evolutia timpului (Gradient vs SGD)');
legend('Gradient', 'SGD');

% TESTARE SI EVALUAREA PERFORMANTEI
N = size(A_test, 1);
A = [A_test ones(N, 1)];   % setul de date pentru testare
performanceTest(e_test, y(A, X1, x1), y(A, X2, x2));