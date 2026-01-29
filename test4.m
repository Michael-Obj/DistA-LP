n = 4;  % number of real locations
m = 4;  % number of reported locations

% Random utility and distance matrices (for demo)
U = rand(n, m);     
D = rand(n, n);     % Make D symmetric and with zeros on diagonal
D = (D + D') / 2;
D(1:n+1:end) = 0;

epsilon = 1.0;

P = data_perturbation(U, D, epsilon);
disp(P);
