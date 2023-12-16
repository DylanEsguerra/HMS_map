HM(:,:,1) = [0.3576	0.6623	0.9802
0.1722	0.2918	0.455
0.132	0.1868	0.2848
0.146	0.2062	0.3008
0.1588	0.2179	0.3102];

HM_sd(:,:,1) = [0.2844	0.4534	0.3103
0.1152	0.2966	0.3664
0.0442	0.0951	0.1352
0.0801	0.1698	0.237
0.0797	0.1438	0.1876];

HM(:,:,2) = [0.2588	0.528	0.8924
0.1586	0.4004	0.8552
0.1363	0.3487	0.621
0.1446	0.3553	0.6356
0.1482	0.3515	0.6521];

HM_sd(:,:,2) = [0.1241	0.1521	0.227
0.0982	0.2887	1.626
0.0442	0.1322	0.2653
0.069	0.1787	0.3847
0.0684	0.1475	0.4902];

% 12% 13.5% .........................................

%HM(:,:,1) = [0.1536	0.233	0.3815
%0.1594	0.233	0.3634
%0.1689	0.2471	0.3717
%0.1429	0.2053	0.3086
%0.145	0.2023	0.317];

%HM_sd(:,:,1) = [0.1193	0.2165	0.3404
%0.0973	0.1678	0.2607
%0.1148	0.2298	0.3474
%0.0693	0.1362	0.1951
%0.0861	0.1303	0.1979];

%HM(:,:,2) = [0.1404	0.3356	0.6449
%0.1483	0.3795	0.7151
%0.1559	0.3921	0.7175
%0.1435	0.3451	0.6282
%0.1409	0.3559	0.6444];

%HM_sd(:,:,2) = [0.0594	0.2165	0.3552
%0.0538	0.1678	0.4843
%0.0842	0.2298	0.4455
%0.0801	0.1362	0.3014
%0.0558	0.1303	0.3512];


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
        
        
        if (j == 1 & i == 2)
            title("Step Ahead Leave k Out")
            legend("HMS-map","FontSize",20)
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



