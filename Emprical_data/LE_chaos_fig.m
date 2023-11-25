smap_lyp = readmatrix('/Users/dylanesguerra/Desktop/Munch/GDPP/LE_mean.csv');
%smap_lyp(16) = [];
hm_lyp = Lyp_all;

figure;
scatter(smap_lyp, hm_lyp,150,"filled");

% Add lines at x = 0 and y = 0
hold on;
xlim([min(smap_lyp)-0.2, max(smap_lyp)]);
ylim([min(hm_lyp)-0.2, max(hm_lyp)]);

line([0 0], ylim, 'Color', [0.6,0.6,0.6],"LineStyle","--","LineWidth",2);  % Vertical line at x = 0
line(xlim, [0 0], 'Color', [0.6,0.6,0.6],"LineStyle","--","LineWidth",2);  % Horizontal line at y = 0

% Set the plot limits to include the lines

set(gca,"FontSize",25);

% Set plot title and axis labels
title('LE comparison',"FontSize",25);
xlabel('S-map',"FontSize",25);
ylabel('HMS-map',"FontSize",25);

% Add labels for the quadrants
%text(0.1, 0.9, 'HMS-map chaotic', 'Units', 'normalized',"FontSize",25);
%text(0.1, 0.1, 'Both non-chaotic', 'Units', 'normalized',"FontSize",25);
%text(0.7, 0.9, 'Both chaotic', 'Units', 'normalized',"FontSize",25);
%text(0.7, 0.1, 'S-map chaotic', 'Units', 'normalized',"FontSize",25);
