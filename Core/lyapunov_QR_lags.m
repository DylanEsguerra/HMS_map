function lambda = lyapunov_QR_lags(coef, T, l)
%gets LE from jacobian of local linear regression
%l = E-1
Il=[eye(l-1) zeros(l-1,1)];
Q=eye(l);
ll=0;
for i=1:T-l % changed to T-l for lags
    J=[coef(1:l,i)';Il];
    [Q,R]=qr(J*Q');
    ll=ll+log(abs(R(1)))/T;
end


lambda = ll;