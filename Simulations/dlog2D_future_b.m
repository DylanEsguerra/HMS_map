function[MSE,Total_MSE,Predictions,X_pred,Emb,Theta,Coefs,X_true,Y_noise,XP] = dlog2D_future_b(n,start,step,burn,obs,smap)

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

%r = [3.8, 3.5]; % Chaotic
r = [3.1, 3.1]; % periodic
B = [-0.02,-0.1];
for i = 1:n 
    x = rand(T,2); 
    for t = 1:T
        x(t+1,1) = x(t,1)*(r(1)-r(1)*x(t,1)-B(1)*x(t,2));
        x(t+1,2) = x(t,2)*(r(2)-r(2)*x(t,2)-B(2)*x(t,1));
    end

   

    x = x(burn:T,2);
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




