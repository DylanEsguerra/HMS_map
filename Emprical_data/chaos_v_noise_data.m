Insects_data = readmatrix('/Users/dylanesguerra/Desktop/Munch/GDPP/Insects.csv');
Insects_data = Insects_data(2:end,2:end); %removes index
%Insects_data(:,16) = [];
n = 16;
step = 3;
seq_obs = 0:0.1:1; %observation noise values 

E_all = zeros(1,n);
Lyp_all = zeros(1,n);
theta_all = zeros(1,n);
noise_all = zeros(1,n);
Y = cell(1,n);
XP = cell(1,n);
X_pred = cell(1,n);

oe_1 = zeros(1,n);
oe_3 = zeros(1,n);
vp = zeros(1,n);

v_I = zeros(1,n);


for i = 1:n
    data = Insects_data(~isnan(Insects_data(:, i)), i);
    data_scaled = data;%normalize(data);
    %data_scaled = (data-mean(data))/std(data);
    Y{i} = data_scaled;
    T = length(data_scaled);

    FNN = f_fnn(data_scaled,1,10,15,2); 
    [~,E]= min(FNN);

    v_I(i) = var(data);
    test = rand(length(seq_obs),1); %saves observation error
    
    theta = zeros(1,length(seq_obs));
    %z = 0;
    for j = 1:length(seq_obs)
        noise = (seq_obs(j)*std(data_scaled))^2;
        fun = @(z)HMSmap_lags(data_scaled,[],'gaussian',z,noise,E-1,1,0,step).oe(step); 
        z = fminbnd(fun,0,50);%finds optimal theta min prediction error
        theta(j) = z;
        test(j) = HMSmap_lags(data_scaled,[],'gaussian',z,noise,E-1,1,0,step).oe(step);
    end

    

    [M,I] = min(test);% gets best value for observation noise 
   
   
    theta_all(i) = theta(I);
    noise_all(i) = seq_obs(I);

    fun = @(z)HMSmap_lags(data_scaled,[],'gaussian',z,0,E-1,1,0,step).oe(step); 
    z = fminbnd(fun,0,50);%finds optimal theta min prediction error
    out = HMSmap_lags(data_scaled,[],'gaussian',z,0,E-1,1,0,step);%theta(I);

    XP{i} = out.states;
    X_pred{i} = out.pred; 

    E_all(i) = E;

    oe_1(i) = sqrt(out.oe(1)/var(data_scaled));
    oe_3(i) = sqrt(out.oe(3)/var(data_scaled));
    vp(i) = sqrt(out.vp/var(data_scaled));
    Lyp_all(i) = lyapunov_QR_lags(out.coef, T-(E-1), E-1);


    if mod(i, 1) == 0
        figure
        hold on
        plot(data_scaled,"LineWidth",2)
        plot(out.states,"LineWidth",2)
        hold off
        legend("Observed","Smoothed","FontSize",20)
        
        ylabel('Population')
        title(['Insect' num2str(i) ')','noise' num2str(seq_obs(I)),"E",num2str(E-1)],"FontSize",20)
    end

end 
%process = noise_all - pe_1; negative 
%process = V_1 - noise_all;
%mean(pred_err,2)
