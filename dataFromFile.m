% citeste o linie din fisier
function [A_learn, A_test, e_learn, e_test] = dataFromFile(N)
    A = [readmatrix('winequality-red.csv'); 
        readmatrix('winequality-white.csv')];

    idx = randperm(size(A, 1));
    A   = A(idx, :);

    nr = round(0.8 * N);    % numarul testelor pentru invatare
    A_learn = A(1:nr, 1:(end - 1));
    A_learn = (A_learn - mean(A_learn)) ./ std(A_learn);
    A_test  = A((nr+1):N, 1:(end - 1));
    A_test  = (A_test - mean(A_test)) ./ std(A_test);

    e_learn = A(1:nr, end);
    e_test  = A((nr+1):N, end);
end