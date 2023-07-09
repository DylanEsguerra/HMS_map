% Numerical Summaries

n = 100; % Number of replications
Noise = zeros(5, n); % Array to store noise values
Obs_err = zeros(5, n); % Array to store observation error values
Filter_err = zeros(5, n); % Array to store filter error values
Noise_m = zeros(5, 1); % Array to store mean noise values
Obs_err_m = zeros(5, 1); % Array to store mean observation error values
Filter_err_m = zeros(5, 1); % Array to store mean filter error values
Noise_sd = zeros(5, 1); % Array to store standard deviation of noise values
Obs_err_sd = zeros(5, 1); % Array to store standard deviation of observation error values
Filter_err_sd = zeros(5, 1); % Array to store standard deviation of filter error values
Lyp = zeros(5, n); % Array to store Lyapunov exponents

obs = [0.05, 0.1, 0.15, 0.2, 0.25]; % Array of noise levels
E = Emb; % Embedding dimensions, data sometimes saved as Emb

for i = 1:5
    for j = 1:n
        Noise(i, j) = sqrt(mean((X_true{i}{j}(2:101) - Y{i}{j}(2:101)).^2) / var(X_true{i}{j}(2:101))); % Calculate noise value
        Obs_err(i, j) = sqrt(mean((XP{i}{j}(2:101) - Y{i}{j}(2:101)).^2) / var(X_true{i}{j}(2:101))); % Calculate observation error value
        Filter_err(i, j) = sqrt(mean((XP{i}{j}(2:101) - X_true{i}{j}(2:101)).^2) / var(X_true{i}{j}(2:101))); % Calculate filter error value
        
        Lyp(i, j) = lyapunov_QR_lags(Coefs{i}{j}, 101 - (E{i}{j} - 1), E{i}{j} - 1); % Calculate Lyapunov exponent
        
        %Var_species = Var_species + var(X_true{i}{j}(:,5));
        
        %Lyp(i, j) = lyapunov_QR_lags(Coefs{i}{j}, 101, 1);
    end 
    
    Noise_m(i) = mean(Noise(i, :)); % Calculate mean noise value
    Obs_err_m(i) = mean(Obs_err(i, :)); % Calculate mean observation error value
    Filter_err_m(i) = mean(Filter_err(i, :)); % Calculate mean filter error value
    
    %Noise_sd(i) = std(Noise(i, :)); % Calculate standard deviation of noise values
    %Obs_err_sd(i) = std(Obs_err(i, :)); % Calculate standard deviation of observation error values
    Filter_err_sd(i) = std(Filter_err(i, :)); % Calculate standard deviation of filter error values
end 


% Graphical Summaries

% figure(1)
hold on
plot(obs, Obs_err_m, '-o', 'MarkerSize', 10, 'LineWidth', 2) % Plot observation error
plot(obs, Noise_m, '-o', 'MarkerSize', 10, 'LineWidth', 2) % Plot noise values
plot(obs, Filter_err_m, '-o', 'MarkerSize', 10, 'LineWidth', 2) % Plot filter error
% errorbar(obs, Filter_err_m, Filter_err_sd, '-o', 'MarkerSize', 10, 'LineWidth', 2) % Plot filter error with error bars
% plot(obs, obs, 'LineWidth', 2)
% set(gca, 'FontSize', 20)
legend("HMS-map Smoothing", "X = Y", 'FontSize', 20)
% xlabel("Observation Noise", 'FontSize', 20)
% ylabel("MSE", 'FontSize', 20)
% title('1D', 'FontSize', 20);
hold off 

figure(2)
hold on
dropplot(Lyp', obs) % Plot Lyapunov exponents
yline(-0.2056, 'r', 'LineWidth', 5) % Add a line at a specific Lyapunov exponent value
%cyclic dlog -0.0999, 2d -0.2056
%3.5 -0.8736
xlabel("Observation Noise", 'FontSize', 20)
ylabel("Lyapunov Exponent", 'FontSize', 20)
title("2D HMS-map Lyapunov Exponent", 'FontSize', 20)
hold off 
