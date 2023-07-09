function [MSE, Total_MSE, Predictions, X_pred, Emb, Theta, Coefs, X_true, Y_noise, XP] = dlog2D_future_b(n, start, step, burn, obs, smap)
% Performs future predictions and calculates MSE for a 2-dimensional system

T = 221; % Time series length

MSE = rand(n, length(start+burn-1:T-step)); % Matrix to store MSE values
Predictions = cell(n, 1); % Cell array to store predictions

XP = cell(n, 1); % Cell array to store smoothed values
Emb = cell(n, 1); % Cell array to store embedding dimensions
Theta = cell(n, 1); % Cell array to store parameters
X_true = cell(n, 1); % Cell array to store true values
Y_noise = cell(n, 1); % Cell array to store noisy observed data
Coefs = cell(n, 1); % Cell array to store coefficients
X_pred = cell(n, 1); % Cell array to store future predictions

% Parameters for the 2-dimensional system
% r = [3.8, 3.5]; % Chaotic
r = [3.1, 3.1]; % Periodic
B = [-0.02, -0.1];

% Generate data from the 2-dimensional system
for i = 1:n 
    x = rand(T, 2); 
    for t = 1:T
        x(t+1, 1) = x(t, 1) * (r(1) - r(1) * x(t, 1) - B(1) * x(t, 2));
        x(t+1, 2) = x(t, 2) * (r(2) - r(2) * x(t, 2) - B(2) * x(t, 1));
    end

    x = x(burn:T, 2);
    y = x + obs * std(x) * randn(length(x), 1); % Adds observational noise

    X_true{i} = x; % Store the true values
    Y_noise{i} = y; % Store the noisy observed data

    trunc_y = y(1:start); % Truncated part of observed data 

    [FNN] = f_fnn(trunc_y, 1, 10, 15, 2); 
    [~, E] = min(FNN); % Estimate embedding dimension using False Nearest Neighbors (FNN)

    obs_err = obs;
    % If smap noise is still added to data, but not seen by algorithm
    if smap
        obs_err = 0;
    end

    noise = (obs_err .* std(x)).^2; % Calculate noise level

    % Estimate theta parameter using optimization (fminbnd) and HMSmap_lags function
    fun = @(z) HMSmap_lags(trunc_y, [], 'gaussian', z, noise, E-1, 1, 0, step).oe(step); 
    z = fminbnd(fun, 0, 50); % Optimize the objective function to estimate theta

    out = HMSmap_lags(trunc_y, [], 'gaussian', z, noise, E-1, 1, 0, step); % Perform HMSmap_lags

    XP{i} = out.states; % Store the smoothed values
    Emb{i} = E; % Store the estimated embedding dimension
    Theta{i} = z; % Store the estimated parameter
    Coefs{i} = out.coef; % Store the coefficients
    X_pred{i} = out.pred; % Store the future predictions

    [MSE(i, :), Predictions{i}] = Future_pred_sim(x, y, noise, step, start, E, z); % Perform future predictions and calculate MSE
end 

Total_MSE = mean(MSE); % Calculate the mean MSE
