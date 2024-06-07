% PLOT PHASE PORTRAIT OF SOLUTIONS
function plot_portraits(p_cell, t_cell)
%     addpath ARFM_Functions\
    row = 0;
colors     	=   {'#1a5276 ','#f39c12 ', '#a3e4d7 '};
    markersizes	=   [24,16,12];
    markers 	=   ['.','.','.'];
    lines 	=   ['1','0.5','0.5'];
    names = {'Truth','Biased filter sol.','Unbiased filter sol.'};
    figure('Units','normalized','OuterPosition',[0 0.06 0.8 0.8]); 
    tiledlayout(2, 1,'Padding','compact','TileSpacing','compact')
    for j = 1:length(p_cell)
        P   =   p_cell{j};
        X	=   p_cell{j};
        T 	=   t_cell{j};
        Fs  =   1 / (T(2) - T(1));           % Sampling frequency
        if j ==1 % Fix same tau and d for all the plots
            try 
            %% average mutual information and get optimum time delay
            lag_max     =   0.05 * Fs; %Maximum lag
            % Average mutual information and lag positions from 'ARFM_ami.m' function
            [v,~]       =   ARFM_ami(X,lag_max);
            % The first local minimum is at the optimal time delay
            [~,tau_AMI]	=   findpeaks(-v);
            Delay    	=   tau_AMI(1);
            zeta     	=   floor(Fs*Delay/1000);    % Transform to seconds 
            %% Obtaining embedding dimension from a False Nearest Neighbor
            % Set typical input values for algorithm
            dmax=10; Rtol=50; Atol=2;
            % Extract the % value of false nearest neighbors from ARFM_fnn.m
            X_ = P(end-2000:end);
            T_ = T(end-2000:end);
            [FNN]=ARFM_fnn(X_,T_,Fs,dmax,zeta,Rtol,Atol);
    
            fnn_zero = find(FNN==0,1,'first');
            d = fnn_zero + 1;
            catch
                d = 3;
                zeta = 12;
            end

        end
        %% Phase space reconstruction
        % Find length of the delay vectors (M) using the values of number 
        % of data points in the signal (N),minimum embedding dimension, (d)
        % and the optimum time delay (tau) as M=N-(d-1)*tau
        M   =   length(X) - (d - 1) * zeta;
        % Find number of delayed vectors of a signal from the given values 
        % of embedding dimension and delay
        [Y]     =   ARFM_delay_vec(X,zeta,d,M);  
        % 3-D reconstructed phase portrait in embedding dimensional space
        ax = nexttile(row+1); 
        if size(Y,2) == 2
            plot(ax,Y(:,1),Y(:,2),'linewidth',lines(j),'color',[0 .5 0]);
            title('2D Phase Portrait')
        else
            plot3(ax,Y(:,1),Y(:,2),Y(:,3),'linewidth',1,'color',colors{j}); 
            zlabel(ax,'$p''(t+2\zeta)$ [Pa]')
        end
        axis(ax,'square'); box(ax, 'on'); grid on; hold on
        xlabel(ax,'$p''(t)$ [Pa]'); 	ylabel(ax,'$p''(t+\zeta)$ [Pa]')   
        %% Plot First Return Map
        max_h = 0;
        [pks,~]  =   findpeaks(X,'MinPeakDistance',10,'Minpeakheight',max_h/2);        
        M 	=   1.2 * max(pks);
        ax  =   nexttile(row+2);	hold on; axis(ax,'square');
        if j ==1
        plot(ax,0:1/Fs:2*M,0:1/Fs:2*M,'linewidth',1,'Color', 'k',...
            'HandleVisibility','off'); grid on; 
        end
        plot(ax,pks(1:end-1),pks(2:end),'LineStyle','none',...
            'marker', markers(j),'Color',colors{j},'MarkerSize',...
            markersizes(j),'DisplayName',names{j});
        xlim(ax,[0,M]);	ylim(ax,[0,M]);     
        xlabel(ax,'$p''_\mathrm{max} (i)$ [Pa]')
        ylabel(ax,'$p''_\mathrm{max} (i+1)$ [Pa]')   
        legend('Location', 'eastoutside','NumColumns', 1,'EdgeColor','None')
        
        if max(pks) <0.01 
            xlim(ax,[0,0.25]);	ylim(ax,[0,0.25]);
        end
    end
end



