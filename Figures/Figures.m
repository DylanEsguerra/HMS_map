%numerical Summaries 
% creates figures for LE and for smoothed error against the added noise
n = 100; %replications
Noise = zeros(5,n);
Obs_err = zeros(5,n);
Filter_err = zeros(5,n);
Noise_m = zeros(5,1);
Obs_err_m = zeros(5,1);
Filter_err_m = zeros(5,1);
Noise_sd = zeros(5,1);
Obs_err_sd = zeros(5,1);
Filter_err_sd = zeros(5,1);
Lyp = zeros(5,n);
Var_species = zeros(1,1);
obs = [0.05,0.1,0.15,0.2,0.25];
E = Emb; %sometimes saved E as Emb
for i = 1:5
    for j = 1:n
        
        Noise(i,j) =  sqrt(mean((X_true{i}{j}(2:101)-Y{i}{j}(2:101)).^2)/var(X_true{i}{j}(2:101)));
        Obs_err(i,j) =  sqrt(mean((XP{i}{j}(2:101)-Y{i}{j}(2:101)).^2)/var(X_true{i}{j}(2:101)));
        Filter_err(i,j) =  sqrt(mean((XP{i}{j}(2:101)-X_true{i}{j}(2:101)).^2)/var(X_true{i}{j}(2:101)));
        %lyapunov exponents
        Lyp(i,j) = lyapunov_QR_lags(Coefs{i}{j}, 101-(E{i}{j}-1), E{i}{j}-1);
        %Var_species = Var_species + var(X_true{i}{j}(:,5));


        %Lyp(i,j) = lyapunov_QR_lags(Coefs{i}{j}, 101, 1);
      
        
 
        
    end 
    
    Noise_m(i) = mean(Noise(i,:));
    Obs_err_m(i) = mean(Obs_err(i,:));
    Filter_err_m(i) = mean(Filter_err(i,:));
    %Noise_sd(i) = std(Noise(i,:));
    %Obs_err_sd(i) = std(Obs_err(i,:));
    Filter_err_sd(i) = std(Filter_err(i,:));
end 

Var_species = Var_species/(n*5);

%graphical Summaries 
%figure(1)
hold on
plot(obs,Obs_err_m,'-o','MarkerSize',10,'LineWidth',2)
plot(obs,Noise_m,'-o','MarkerSize',10,'LineWidth',2)
plot(obs,Filter_err_m,'-o','MarkerSize',10,'LineWidth',2)
%errorbar(obs,Filter_err_m,Filter_err_sd,'-o','MarkerSize',10,'LineWidth',2)
%plot(obs,obs,'LineWidth',2)
%set(gca,"FontSize",20)
legend("HMS-map Smoothing","X = Y",'FontSize',20)
%xlabel("Observation Noise",'FontSize', 20)
%ylabel("MSE",'FontSize', 20)
%title('1D', 'FontSize', 20);
hold off 


figure(2)
hold on
dropplot(Lyp',obs)
yline(-0.2056,'r','LineWidth',5)  %dlog yline(0.693); %henon 0.419222 %2d 0.42  %5d 0.4784
%cyclic dlog -0.0999, 2d -0.2056
%3.5 -0.8736
xlabel("observation noise", 'FontSize', 20)
ylabel("Lyapunov Exponent", 'FontSize', 20)
title("2D HMS-map Lyapunov Exponent", 'FontSize', 20)
hold off 
