%calculating error for future predictions from saved future pred data

n = 100; %number of iterations 
T = 122; %full time series length 
start = 101;
step = 3;
N = length(start:T-3);

MSE_mat = zeros(5,N,n);
MSE_mat_m = zeros(5,N);
MSE_mat_sd = zeros(5,N);
for i = 1:5
    for j = 1:n
        data = X_true{i}{j};
        
        for k = start:T-3
            idx = k +1 - start;
            pred = Predictions{i}{j}{idx};
            trunc = Y{i}{j}(1:k);
            T_t = length(trunc);
            step_err = mean((data(T_t:T_t+(step-1))-pred(T_t-(step-1):T_t,step)).^2);
            
            MSE_mat(i,idx,j) = sqrt(step_err/var(data(1:k)));
           
        end 
        
    end 
  
end 

MSE_mat_m = mean(mean(MSE_mat,2),3)
MSE_mat_sd = std(mean(MSE_mat,2),0,3)

%figure;
%hold on;
%plot((180:198),MSE_mat_m,"LineWidth",2)
%legend("0.05","0.1","0.15","0.2","0.25","FontSize",15)
%title("HMSMAP 3 step 5D model","FontSize",20)
%ylabel("SQRT MSE","FontSize",20)
%legend("0.05","0.1","0.15","0.2","0.25","FontSize",15)
%hold off;

