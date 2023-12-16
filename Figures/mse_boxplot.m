figure;

rows = 1;
cols = dim1;

for i = 1:dim1
    subplot(rows, cols, i);
    hold on;
    for j = 1:dim2
        bh = boxplot(squeeze(squeeze(MSE_all(i, j, :))), 'positions', j, 'widths', 0.8, 'symbol', '');
        ylim([0 1.3]);
        set(bh,'LineWidth', 4);
    end
    
    title(['Step: ' num2str(i)], 'FontSize', 25);
    if i == 2
        xlabel('Observed Observation Noise', 'FontSize', 25);
    end 
    ylabel('NRMSE', 'FontSize', 25);
    xticks(1:dim2);
    %xticklabels({'12%', '13.5%', '15%', '16.5%', '18%'});
    xticklabels({'5%', '10%', '15%', '20%', '25%'});
    %yticklabels({'0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%'});
    yticklabels({'0%', '20%', '40%', '60%', '80%', '100%', '120%'});
    set(gca, 'FontSize', 20);
    hold off;
end


