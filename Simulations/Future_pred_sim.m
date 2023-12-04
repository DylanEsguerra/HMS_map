function [MSE, Predictions] = Future_pred_sim(X, Y, noise, step, start, E, Theta)
% Performs future forecasts on test data.
%
% Inputs:
%   X:      Full dataset.
%   Y:      Observed data.
%   noise:  Noise level.
%   step:   Number of future steps to predict.
%   start:  Starting index for test data.
%   E:      Embedding dimension from FNN.
%   Theta:  Local weighting Parameter for the HMSmap_lags function.
%
% Outputs:
%   MSE:         Mean squared error for each prediction.
%   Predictions: Cell array of predicted values for each test data point.

N = length(X); % Full dataset length

N_S = length(start:(N-step)); % Full truncated length
MSE = rand(1, N_S);
Predictions = cell([N_S, 1]);

inits = [];

for i = start:N-step % Loop over test data
    idx = i + 1 - start;

    trunc = Y(1:i); % Truncated part of observed data

    T = length(trunc);

    % Perform future forecast using HMSmap_lags_b function
    out = HMSmap_lags(trunc, 'gaussian', Theta, noise, E-1, 1, 0, step, inits);
    inits = out.states;
    inits(length(trunc)+1) = Y(i + 1);

    Predictions{idx} = out.pred;
    
    % Calculate mean squared error (MSE)
    MSE(idx) = mean((X(T:T+(step-1)) - out.pred(T-(step-1):T, step)).^2);
    MSE(idx) = sqrt(MSE(idx) / var(X(1:i)));

end


