function [MSE, Predictions] = Future_pred_sim(X,Y,noise,step,start,E,Theta)
%does future forecasts on test data

N = length(X); % Full dataset length

N_S = length(start:(N-step)); % full truncated length
MSE = rand(1,N_S);
Predictions = cell([N_S,1]);

inits = [];

for i = start:N-step %loop over test data 
    idx = i + 1 - start;

    trunc = Y(1:i); %truncated part of observed data 
    

    T = length(trunc); 

    
    
    out = HMSmap_lags_b(trunc,[],'gaussian',Theta,noise,E-1,1,0,step,inits);
    inits = out.states;
    inits(length(trunc)+1) = Y(i + 1);

   
    
    Predictions{idx} = out.pred;
    MSE(idx) = mean((X(T:T+(step-1))-out.pred(T-(step-1):T,step)).^2);
    MSE(idx) = sqrt(MSE(idx)/var(X(1:i))); 

end 

