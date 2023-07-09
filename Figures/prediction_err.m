% Calculates prediction error for leave-one-out on saved data
% filter error is smooth vs true data while prediction error is the leave
% one out predictions against the true data
n = 100; % Number of repetitions
T = 101; % Time index
pred_err = zeros(5, 3, n); % Array to store prediction error
pred_err_m = zeros(5, 3); % Array to store mean prediction error
pred_err_sd = zeros(5, 3); % Array to store standard deviation of prediction error
Filter_err = zeros(5, n); % Array to store filter error
Filter_err_m = zeros(5, 1); % Array to store mean filter error
Filter_err_sd = zeros(5, 1); % Array to store standard deviation of filter error
Var_species = zeros(1, 1); % Array to store species variance
obs = [0.05, 0.1, 0.15, 0.2, 0.25]; % Array of noise levels
E = Emb; % Embedding dimensions

for i = 1:5
    for j = 1:n
        
        E_1 = E{i}{j} - 1; % Get embedding dimension minus 1
        
        Filter_err(i, j) = sqrt(mean((XP{i}{j}(2:T) - X_true{i}{j}(2:T)).^2) / var(X_true{i}{j}(2:T))); % Calculate filter error
        
        for k = 1:3
            pred_err(i, k, j) = sqrt(mean((X_pred{i}{j}(E_1+1:T-(k-1), k) - X_true{i}{j}(E_1+k:T)).^2) / var(X_true{i}{j}(E_1+k:end))); % Calculate prediction error
        end
    end
    
    pred_err_m(i, :) = mean(pred_err(i, :, :), 3); % Calculate mean prediction error
    pred_err_sd(i, :) = std(pred_err(i, :, :), 0, 3); % Calculate standard deviation of prediction error
    Filter_err_m(i) = mean(Filter_err(i, :)); % Calculate mean filter error
    Filter_err_sd(i) = std(Filter_err(i, :)); % Calculate standard deviation of filter error
end

Filter_err_m % Display mean filter error

pred_err_m(:, 1) % Display mean prediction error for lag 1
pred_err_m(:, 2) % Display mean prediction error for lag 2
pred_err_m(:, 3) % Display mean prediction error for lag 3

Filter_err_sd % Display standard deviation of filter error

pred_err_sd(:, 1) % Display standard deviation of prediction error for lag 1
pred_err_sd(:, 2) % Display standard deviation of prediction error for lag 2
pred_err_sd(:, 3) % Display standard deviation of prediction error for lag 3
