function[X,stdp]=lag(x,d,tau,standardize);
n=length(x);
xx=[];stdp.m=0;stdp.s=1;
if standardize==1,%0 mean, sd 1
    stdp.m=mean(x,'omitnan');stdp.s=std(x,'omitnan');
    x=(x-stdp.m)/stdp.s;
end

if standardize==3,%scaled to max=1
    ma=max(xx,'omitnan');
    stdp.m=0;stdp.s=ma;
    xx=(xx-stdp.m)/stdp.s;
end


for j=1:d, 
    xx=[x([1:n-(d-1)*tau]+(j-1)*tau) xx];
end

if standardize==2, %each column shifted and scaled into [0,1]
    mi=min(xx,[],'omitnan');ma=max(xx,[],'omitnan');
    stdp.m=mi;stdp.s=(ma-mi);
    xx=(xx-mi)*diag(1./(ma-mi));
end

X=xx;