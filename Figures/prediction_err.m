%calculates prediction error for leave one out on saved data 
n = 100;
T = 101;
pred_err = zeros(5,3,n);
pred_err_m = zeros(5,3);
pred_err_sd = zeros(5,3);
Filter_err = zeros(5,n);
Filter_err_m = zeros(5,1);
Filter_err_sd = zeros(5,1);
Var_species = zeros(1,1);
obs = [0.05,0.1,0.15,0.2,0.25];
E = Emb;
for i = 1:5
    for j = 1:n

        E_1 = E{i}{j} - 1;

    
        Filter_err(i,j) =  sqrt(mean((XP{i}{j}(2:T)-X_true{i}{j}(2:T)).^2)/var(X_true{i}{j}(2:T)));
        for k = 1:3
           
            pred_err(i,k,j) =  sqrt(mean((X_pred{i}{j}(E_1+1:T-(k-1),k)-X_true{i}{j}(E_1+k:T)).^2)/var(X_true{i}{j}(E_1+k:end)));
        end 
        
    end
    pred_err_m(i,:) = mean(pred_err(i,:,:),3);
    pred_err_sd(i,:) = std(pred_err(i,:,:),0,3);
    Filter_err_m(i) = mean(Filter_err(i,:));
    Filter_err_sd(i) = std(Filter_err(i,:));
end 
Filter_err_m


pred_err_m(:,1)
pred_err_m(:,2)
pred_err_m(:,3)

Filter_err_sd

pred_err_sd(:,1)
pred_err_sd(:,2)
pred_err_sd(:,3)



