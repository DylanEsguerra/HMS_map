function[]=dropplot(data,x,color);
%data is a matrix of observations organized into columns;
%x is a row vector of column labels for data

% l=min(data);
% u=max(data);
% [n,c]=size(data);
% sd=sort(data);
% z=sd(round([.25;.75]*n),:);
% 
% c=length(x);
% 
% dx=min(abs(diff(x)));if length(x)==1, dx=1;end
% w=.3*dx;
% if nargin<3,
%     color=['k'];
% end
% for i=1:c,
%     [f,q]=hist(data(:,i),min(30,n/2));
%     f=f'/max(f)*w;
%     box=[x(i)+f q';
%         flipud(x(i)-f) flipud(q')];
%     %drop
%     patch(box(:,1),box(:,2),[.75 .75 .75]);hold on;
%     plot(box(:,1),box(:,2),color);
%     %mean line
%     plot(x(i)+w*[-1 1], mean(data(:,i))*[1 1],color,'linewidth',3); 
% end
% 
% set(gca,'XTick',x)
% 

l=min(data);
u=max(data);
[n,c]=size(data);
sd=sort(data);
z=sd(round([.25;.75]*n),:);

c=length(x);

dx=min(abs(diff(x)));if length(x)==1, dx=1;end
w=.3*dx;
if nargin<3,
    color=ones(c,1)*[.75 .75 .75];
end
for i=1:c,
    [f,q]=hist(data(:,i),min(30,n/2));
    f=f'/max(f)*w;
    box=[x(i)+f q';
        flipud(x(i)-f) flipud(q')];
    %drop
    patch(box(:,1),box(:,2),color(i,:));hold on;
    plot(box(:,1),box(:,2),'k','linewidth',1);
    %mean line
    plot(x(i)+w*[-1 1], mean(data(:,i))*[1 1],'k','linewidth',3); 
%    plot(x(i)+w*[-1 1], median(data(:,i))*[1 1],'k','linewidth',3); 
end

set(gca,'XTick',x)