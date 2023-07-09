% Main file
% Calls the specified simulated model and gets smoothed values XP as well
% as future predictions X_pred at specified noise levels

Total_MSE = cell([5, 1]); % Cell array to store total MSE for each noise level
MSE = cell([5, 1]); % Cell array to store MSE for each repetition and noise level
Predictions = cell([5, 1]); % Cell array to store future predictions for each repetition and noise level
Emb = cell([5, 1]); % Cell array to store embedding dimensions for each noise level
Theta = cell([5, 1]); % Cell array to store local weight parameters for each noise level
Y = cell([5, 1]); % Cell array to store observed data for each noise level
XP = cell([5, 1]); % Cell array to store smoothed values for each repetition and noise level
X_pred = cell([5, 1]); % Cell array to store leave one out predictions for each repetition and noise level
Coefs = cell([5, 1]); % Cell array to store coefficients for each repetition and noise level
X_true = cell([5, 1]); % Cell array to store true values for each repetition and noise level

obs = [0.05, 0.1, 0.15, 0.2, 0.25]; % Array of noise levels
n = 100; % Number of repetitions
start = 101; % Starting index for future forecasts
step = 3; % Number of future steps to predict
burn = 100; % Burn-in to reach attractor
smap = 1; % If 0, uses HMSmap

% Collect the raw data for each noise level
for i = 1:5
    % Call the dlog_future function to get the results
    [MSE{i}, Total_MSE{i}, Predictions{i}, X_pred{i}, Emb{i}, Theta{i}, Coefs{i}, X_true{i}, Y{i}, XP{i}] = dlog_future(n, start, step, burn, obs(i), smap);
    i
end

