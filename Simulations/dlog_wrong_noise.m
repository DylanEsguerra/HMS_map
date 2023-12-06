function[MSE,Total_MSE,Predictions,X_pred,Emb,Theta,Coefs,X_true,Y_noise,XP] = dlog_wrong_noise(n,start,step,burn,obs,smap)

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

% Data from Discrete Logistic
r = 4; % Logistic map parameter
for i = 1:n
    x = rand(T, 1);
    for t = 1:(T-1)
        x(t+1) = r * x(t) * (1 - x(t));
    end
    
    % Drop the burn-in
    x = x(burn:T);
    y = x + 0.15 * std(x) * randn(length(x), 1); % Adds observational noise always 15%

    X_true{i} = x;
    Y_noise{i} = y;

    trunc_y = y(1:start); % Truncated part of observed data

    [FNN] = f_fnn(trunc_y, 1, 10, 15, 2); 
    [~, E] = min(FNN); % Estimate embedding dimension using False Nearest Neighbors (FNN)

    obs_err = obs;
    % If smap noise is still added to data, but not seen by algorithm
    if smap
        obs_err = 0;
    end

    noise = (obs_err .* std(x)).^2; % Calculate noise level which can be wrong

    % Estimate theta parameter using optimization (fminbnd) and HMSmap_lags function
    fun = @(z) HMSmap_lags(trunc_y, 'gaussian', z, noise, E-1, 1, 0, step,[]).oe(step);
    z = fminbnd(fun, 0, 50); % Optimize the objective function to estimate theta

    out = HMSmap_lags(trunc_y, 'gaussian', z, noise, E-1, 1, 0, step,[]); % Perform HMSmap_lags

    XP{i} = out.states; % Store the smoothed values
    Emb{i} = E; % Store the estimated embedding dimension
    Theta{i} = z; % Store the estimated parameter
    Coefs{i} = out.coef; % Store the coefficients
    X_pred{i} = out.pred; % Store the future predictions

    [MSE(i, :), Predictions{i}] = Future_pred_sim(x, y, noise, step, start, E, z); % Perform future predictions and calculate MSE
end

Total_MSE = mean(MSE); % Calculate the mean MSE
