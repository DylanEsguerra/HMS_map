% Calculating error for future predictions from saved future pred data

n = 100; % Number of iterations
T = 122; % time series length after burn
start = 101; % Starting index for future forecasts
step = 3; % Number of future steps to predict
N = length(start:T-3); % Number of future prediction points

MSE_mat = zeros(5, N, n); % Matrix to store MSE for each noise level, future prediction point, and iteration
MSE_mat_m = zeros(5, N); % Matrix to store mean MSE for each noise level and future prediction point
MSE_mat_sd = zeros(5, N); % Matrix to store standard deviation of MSE for each noise level and future prediction point

for i = 1:5
    for j = 1:n
        data = X_true{i}{j}; % True data for current iteration
        
        for k = start:T-3
            idx = k + 1 - start; % Index for storing MSE
            
            pred = Predictions{i}{j}{idx}; % Future predictions for current future prediction point
            
            trunc = Y{i}{j}(1:k); % Truncated part of observed data
            T_t = length(trunc); % Length of truncated data
            
            % Calculate MSE for current future prediction point
            step_err = mean((data(T_t:T_t+(step-1)) - pred(T_t-(step-1):T_t, step)).^2);
            
            MSE_mat(i, idx, j) = sqrt(step_err / var(data(1:k))); % Store the calculated MSE
        end
    end
end

MSE_mat_m = mean(mean(MSE_mat, 2), 3); % Calculate mean MSE for each noise level and future prediction point
MSE_mat_sd = std(mean(MSE_mat, 2), 0, 3); % Calculate standard deviation of MSE for each noise level and future prediction point

% Display mean MSE and standard deviation for each noise level and future prediction point
MSE_mat_m
MSE_mat_sd

% Plotting the mean MSE
% figure;
% hold on;
% plot((180:198), MSE_mat_m, "LineWidth", 2)
% legend("0.05", "0.1", "0.15", "0.2", "0.25", "FontSize", 15)
% title("HMSMAP 3 step 5D model", "FontSize", 20)
% ylabel("SQRT MSE", "FontSize", 20)
% legend("0.05", "0.1", "0.15", "0.2", "0.25", "FontSize", 15)
% hold off;
