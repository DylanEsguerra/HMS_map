function lambda = lyapunov_QR_lags(coef, T, l)
% Calculates the Lyapunov exponent from the Jacobian of local linear regression.
% Inputs:
%   coef: Matrix of coefficients from the local linear regression. Each column represents
%         the coefficients of a different time step.
%   T:    Total number of time steps.
%   l:    Number of lags used in the local linear regression (E-1).

Il = [eye(l-1) zeros(l-1,1)]; % Identity matrix with the last column zeroed out.
Q = eye(l); % Initialize Q as the identity matrix.
ll = 0; % Initialize the Lyapunov exponent variable.

for i = 1:T-l % Iterate from 1 to T-l 
    J = [coef(1:l,i)'; Il]; % Jacobian matrix of local linear regression.
    [Q, R] = qr(J * Q'); % QR factorization of J * Q'.
    ll = ll + log(abs(R(1))) / T; % Update the Lyapunov exponent.
end

lambda = ll; % Return the Lyapunov exponent.