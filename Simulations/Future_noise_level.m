% Main file 
%calls the speciefied simulated model and gets smoothed values XP as well
%as future predictions X_pred at specified noise levels 
Total_MSE = cell([5,1]);
MSE = cell([5,1]);
Predictions = cell([5,1]);
Emb = cell([5,1]);
Theta = cell([5,1]);
Y = cell([5,1]);
XP = cell([5,1]);
X_pred = cell([5,1]);
Coefs = cell([5,1]);
X_true = cell([5,1]);
obs = [0.05,0.1,0.15,0.2,0.25];
n = 100; % repetitions 
start = 101; % when to start future forecasts 
step = 3;
burn = 100; % burn in to reach attractor 
smap = 1; % if 0 uses HMSmap 

%collect the raw data 
for i = 1:5
    [MSE{i},Total_MSE{i},Predictions{i},X_pred{i},Emb{i},Theta{i},Coefs{i},X_true{i},Y{i},XP{i}] = dlog_future(n,start,step,burn,obs(i),smap);
    i
end

%figure;
%hold on;
%plot((start:198),Total_MSE{1},"LineWidth",2)
%plot((start:198),Total_MSE{2},"LineWidth",2)
%plot((start:198),Total_MSE{3},"LineWidth",2)
%plot((start:198),Total_MSE{4},"LineWidth",2)
%plot((start:198),Total_MSE{5},"LineWidth",2)
%legend("0.05","0.1","0.15","0.2","0.25","FontSize",15)
%hold off;