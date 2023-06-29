function[MSE,Total_MSE,Predictions,X_pred,Emb,Theta,Coefs,X_true,Y_noise,XP] = dlog5D_future_b(n,start,step,burn,obs,smap)

T = 221;


MSE = rand(n,length(start+burn-1:T-step));
Predictions = cell(n,1);


XP = cell(n,1);
Emb = cell(n,1);
Theta = cell(n,1);
X_true = cell(n,1);
Y_noise = cell(n,1);
Coefs = cell(n,1);
X_pred = cell(n,1);
%data from Discrete Logistic
for i = 1:n 
    x = rand(T,5); 
    for t = 1:T
        x(t+1,1) = max(1e-10,x(t,1)*(4 - 4*x(t,1) - 2*x(t,2) - 0.4*x(t,3)));
        x(t+1,2) = max(1e-10,x(t,2)*(3.1 - 0.31*x(t,1) - 3.1*x(t,2) - 0.93*x(t,3)));
        x(t+1,3) = max(1e-10,x(t,3)*(2.12 + 0.636*x(t,1) + 0.636*x(t,2) - 1.21*x(t,3)));
        x(t+1,4) = max(1e-10,x(t,4)*(3.8 - 0.111*x(t,1) - 0.011*x(t,2) + 0.131*x(t,3) - 3.8*x(t,4)));
        x(t+1,5) = max(1e-10,x(t,5)*(4.1 - 0.082*x(t,1) - 0.111*x(t,2) - 0.125*x(t,3) - 4.1*x(t,5)));
    end


    x = x(burn:T,5);
    y = x + obs*std(x)*randn(length(x),1); %adds observational noise

    X_true{i} = x;
    Y_noise{i} = y;

    trunc_y = y(1:start); %truncated part of observed data 



    [FNN] = f_fnn(trunc_y,1,10,15,2); 
    [~,E]= min(FNN);
    
    

    obs_err = obs;
    %If smap noise is still added to data, but not seen by algorithm
    if smap
        obs_err = 0;
    end

    noise = (obs_err.*std(x)).^2;
    
    %theta 
    fun = @(z)HMSmap_lags(trunc_y,[],'gaussian',z,noise,E-1,1,0,step).oe(step); 
   
    z = fminbnd(fun,0,50);

    out = HMSmap_lags(trunc_y,[],'gaussian',z,noise,E-1,1,0,step);
    
    XP{i} = out.states;
    Emb{i} = E;
    Theta{i} = z;
    Coefs{i} = out.coef;
    X_pred{i} = out.pred;

    [MSE(i,:),Predictions{i}] = Future_pred_sim(x,y,noise,step,start,E,z);
end 
Total_MSE = mean(MSE);



