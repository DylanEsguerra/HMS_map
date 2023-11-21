function[out,Xtrack]=HMSmap_lags(x,kernel,theta,vobs,E,tau,figs,stepsahead,inits)
%HMSmap_lags(x,model,kernel,theta,vobs,E,tau,figs,stepsahead,WW)
%this script estimates states for an Smap model in delay coordinates with observation and
%process uncertainty using an EM algorithm
%tau (step size) always set to 1.
%kernel specifies which weighting scheme to use. Choices are exponential and Gaussian
%theta is inverse length scale - set to 0 for global linear model
%vobs is observation variance - set to 0 for model without observation error


%distances are scaled to max for all points

switch kernel
    case 'exponential'
        K=@(D) exp(-theta*D);
    case 'gaussian'
        K=@(D) exp(-theta^2*(D).^2);
    case 'inversedistance'
        K=@(D) 1./(1+theta^2*(D).^2);
    case 'quartic'
        K=@(D) exp(-theta^4*(D).^4);

end


T=length(x);
if isempty(inits), xp=x; else xp = inits; end 
if isempty(inits), xpred=x; else xpred = inits; end 
maxD=sqrt(E)*(max(x)-min(x));
n=T-E*tau; 


%stopping conditions
niter=100;tol=.01;del=10;iter=0;


%variable definitions
It=speye(T);In=eye(n);
B=[eye(E) zeros(E,n)];%skewed stack of regression coefficients 
%B=[B;zeros(n,T)]; 
B=sparse([B;zeros(n,T)]); 
A=zeros(T,1);
bj=zeros(E+1,n);%matrix of regression coefficients
Z=[eye(E) zeros(E,n)];%smoother matrix
Xtrack=[];
o_n=ones(n,1);

%loop to convergence
while del>tol & iter<niter
    iter=iter+1;
    
    %lags of x 
    xx=lag(xp,E+1,tau,0);  %calls lag function %1 column for each E+1
    H=[xx(:,2:end) ones(n,1)]; %2nd column and those after. last one is intercept

    xj=xx(:,1); %first column

  
    %optimize b given x
    vp=0;
    for j=1:n  %T-E columns 
        dj=sqrt(sum((H(j,:)-H).^2,2))/maxD;
        wj=K(dj-min(dj(dj>0)));
        wj(j)=0;wj=wj/sum(wj);

        [U,S,V]=svd(wj.*H); %singular value decomposition 
        iS=S;for k=1:size(S,2), iS(k,k)=double(S(k,k)>1e-5)/(S(k,k)+double(S(k,k)<1e-5));end
        bj(:,j)=V*(iS)'*U'*(wj.*xj);  %*******************************************
        
        bs=sum(bj(:,j));
        if (isinf(bs)|isnan(bs)), bj(:,j)=[xj(j) 1]';end
        
        vp=vp+(xj(j)-H(j,:)*bj(:,j))^2/n; %process noise estimate  
        
        B(E+j,j+[1:E]-1)=bj(E:-1:1,j)';%stack of b's for projection matrix
    end
    vrat=vobs/vp; 
    
    %compute expected value of x given b and y  
    xold=xp;
    A(E+1:T)=bj(E+1,:)';%intercepts
    Q=(It-B);if any(isinf(Q(:)))|any(isnan(Q(:))), keyboard;end
    Ax=vrat*Q'*A; 
    
    %*********************************************************************
    xp=(It+vrat*Q'*Q)\(x+Ax);
    xp=xp*mean(xold)/mean(xp);
    %xp=(xp-mean(xp)+mean(x))*std(x)/std(xp);%maintain center and scale

    
    Xtrack(:,iter)=xp;
    del=max(abs(xp-xold));

    
end

%step-ahead predictions for final state estimates (should be same as xpred=B*xp)
for j=1:n
    xpred(E+j,1)=H(j,:)*bj(:,j);
end
pred_err=mean((x(E+1:T)-xpred(E+1:T)).^2);
obs_err=mean((x-xp).^2);

if stepsahead>1, %iterative prediction
    Hahead=H;
    for k=2:stepsahead
        %update vector
        Hahead=[xpred(E+1:T,k-1) Hahead(:,1:E-1) o_n];
        for j=1:n    
            %find distances/weights
            dj=sqrt(sum((Hahead(j,:)-H).^2,2))/maxD;
            wj=K(dj-min(dj(dj>0)));
            Wja=wj/sum(wj);
            
            [U,S,V]=svd(Wja.*H);
            iS=S;for l=1:size(S,2), iS(l,l)=double(S(l,l)>1e-5)/(S(l,l)+double(S(l,l)<1e-5));end
            bja=V*iS'*U'*(Wja.*xj);
 
            %make prediction
            xpred(E+j,k)=Hahead(j,:)*bja;
        end
        obs_err(k)=mean((x(E+k:T)-xpred(E+1:T-(k-1),k)).^2);
        pred_err(k)=mean((xp(E+k:T)-xpred(E+1:T-(k-1),k)).^2);

    end
end


%outputs and figures
out.states=xp;
out.pred=xpred;
out.coef=bj;
out.pe=pred_err;
out.oe=obs_err;

if figs==1,
    % figure(1);plot(1:iter,Xtrack);
   %  figure(2);
  %   subplot(2,2,1);plot(xp,x,'b.');
 %    subplot(2,2,2);plot(x(E:T-1),xpred(E+1:T),'k.',xp(E:T-1),xp(E+1:T)+.2,'b.',x(E:T-1),x(E+1:T),'r.');
%     subplot(2,1,2);plot([1:T],x,'r.',[1:T],xp,'b'); text(1,.9*max(x),num2str([pred_err obs_err]))
     
%     figure(3);
figure;
    for k=1:stepsahead
        subplot(2,stepsahead,k);plot(x(E:T-k),x(E+k:T),'r.',xp(E:T-k),xp(E+k:T),'b.')
        subplot(2,stepsahead,stepsahead+k);plot(x(E+k:T),xp(E+k:T),'k.',x(E+k:T-1),xpred(E+1:T-k,k),'g.')
    end

end