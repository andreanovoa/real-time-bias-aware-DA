
plot_settings
colors     	=   {'#1a5276 ','#f39c12 ', '#a3e4d7 '};

figure('Units','normalized','OuterPosition',[0 0.06 1 0.95]); 
tiledlayout(2,2,'Padding','compact','TileSpacing','compact')

idx = 3000;
%% ===================== POWER SPECTRTAL DENSITY ===================== %
f4=nexttile(4); hold on
%
[f,PSD_]    =   fun_PSD(dt,p_true(1:idx));
plot(f, PSD_,'color', colors{1},'LineWidth',3); 
[f,PSD_]    =   fun_PSD(dt,p_bias(1:idx));
plot(f, PSD_,'color', colors{2},'LineWidth',1.5);
if p_unbias
    [f,PSD_]    =   fun_PSD(dt,p_unbias(1:idx));
    plot(f, PSD_,'color', colors{3},'LineWidth',1.5);
end
ylabel('PSD'); xlabel('Frequency [Hz]'); 
xlim([0,300]); yl = get(gca,'YLim');
text(5,yl(2)*0.9, ['$\tilde{t} \in [',num2str(t(1),2),', ',...
     num2str(t(end),2),']$ s'],'HorizontalAlignment','left')
l=legend('Truth','Biased filter sol.','Unbiased filter sol.');
l.EdgeColor = 'none';

%% ============= PHASE PORTRAIT AND FIRST RETURN MAP ================ %
p_cell = {p_true(1:idx),p_bias(1:idx)};
t_cell = {t(1:idx), t(1:idx)};
if p_unbias
    p_cell{3} = p_unbias(1:idx);
    t_cell{3} = t(1:idx);
end
plot_portraits(p_cell,t_cell)
