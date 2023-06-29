function[MSE,Total_MSE,Predictions,X_pred,Emb,Theta,Coefs,X_true,Y_noise,XP] = dlog_future(n,start,step,burn,obs,smap)

T = 221; % Time Series length


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
r=4; %3.55
for i = 1:n 
    x=rand(T,1);
    for t = 1:(T-1)
        x(t+1) = r*x(t)*(1-x(t));
    end
    
    %drop the burn in 
    x = x(burn:T);
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


