HM(:,:,1) = [0.0568	0.092	0.1682
0.1066	0.1648	0.2725
0.1385	0.1948	0.3024
0.1926	0.2767	0.395
0.2118	0.2793	0.3902];

HM_sd(:,:,1) = [0.0409	0.0923	0.1432
0.0598	0.1246	0.1859
0.0477	0.0936	0.1433
0.1109	0.224	0.2398
0.1002	0.1553	0.2007];

SM(:,:,1) = [1.0112	0.9854	0.9911
1.0198	0.9933	0.9974
1.0149	0.9877	0.9924
1.0165	0.9882	0.993
1.0147	0.9876	0.9934];

SM_sd(:,:,1) = [0.0235	0.0216	0.0196
0.0227	0.0177	0.0174
0.0245	0.0156	0.0158
0.025	0.0209	0.0206
0.0226	0.0191	0.0179];

HM(:,:,2) = [0.0534	0.1519	0.3156
0.1009	0.2584	0.4864
0.1418	0.3607	0.6618
0.1878	0.4663	0.8065
0.2128	0.4667	0.7845];

HM_sd(:,:,2) = [0.0177	0.074	0.1791
0.0554	0.1174	0.2188
0.0658	0.1623	0.3375
0.0749	0.2318	0.4271
0.0575	0.1373	0.2857];

SM(:,:,2) = [0.9201	0.9793	0.9941
0.888	0.9486	0.9632
0.913	0.9713	0.9855
0.9117	0.9714	0.9853
0.9119	0.9734	0.9861];

SM_sd(:,:,2) = [0.0906	0.0719	0.0691
0.0829	0.0751	0.0775
0.0883	0.0728	0.0743
0.0913	0.0908	0.0895
0.0879	0.0826	0.0815];



obs = [0.05,0.1,0.15,0.2,0.25];
obs_shift = obs +0.005;
% Create a new figure window that will contain the subplots
myfig = figure('Color', 'white');


% Create subplots within the new figure window
for i = 1:3
    for j = 1:2
        subplot(2, 3, (j-1)*3+i);
        %subplot('Position',[0.04+(i-1)/3, 1.025-j/3, 0.28, 0.28]);
        %plot(obs,HM(:,i,j),'-o','MarkerSize',10,'LineWidth',3);
        hold on;
        errorbar(obs,HM(:,i,j),HM_sd(:,i,j),'-o','MarkerSize',10,'LineWidth',2)
        errorbar(obs_shift,SM(:,i,j),SM_sd(:,i,j),'-o','MarkerSize',10,'LineWidth',2)
        %plot(obs,SM(:,i,j),'-o','MarkerSize',10,'LineWidth',3)
        if (j == 1 & i == 2)
            title("Step Ahead Leave k Out")
            legend("HMS-map","Kalman Filter","FontSize",20)
        end 

        if (i == 2 & j == 2)
            title("Future Predictions k Steps Ahead")
        end 

        %title(sprintf("Steps Ahead %d, %dD", i, species(j)),"FontSize",20);
        ax = gca; 
        ax.FontSize = 20; 
        xlabel("Observation Noise","FontSize",20)
        ylabel("MSE","FontSize",20)
        xlim([0.05 0.26])
    end
end
set(findall(myfig, 'Type', 'Text'),'FontWeight', 'Normal')
hold off



