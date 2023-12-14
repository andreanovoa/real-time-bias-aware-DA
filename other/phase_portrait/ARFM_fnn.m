function [FNN]=ARFM_fnn(x,t,Fs,dmax,tao,Rtol,Atol)
% This function calculates the minimum value of embedding dimension required
% for the reconstruction of phases space of a given time sampled data.
% It is based on the concept of false nearest neighbors.
% The main idea is to eliminate the false crossings of the phase space trajectories that mainly arise
% due to the projection of higher-dimensional original attractor into a lower-dimensional phase space
% The method of FNN measures percentage of closeness, in terms of Euclidian distances, of neighboring points
% of the trajectory in a d-dimensional space, and compared it with the d+1 dimensional space.
% If the ratio of these distances is greater than a predefined threshold due to change in the dimension,
% the neighbors of the trajectory are considered as false neighbors
% The minimum embedding dimension is chosen as the next dimension to the value where the percentage of 
% false nearest neighbors become zero for the first time 
%_______________ Inputs _______________
% x:    		Time series
% Fs:   		Sampling Frequency
% dmax:         Maximum number of dimensions that need to be tested
% tao:  		Time-delay required for phase space construction. It is calculated from Average Mutual Information
% Rtol: 		Distance threshold to decide false neighbor (Value lies between 10 < Rtol < 50, usually fixed to 10)
% Atol: 		Another criterion to remove false neighbors (Fixed value, 2) (Abarbanel et al. 1993)
%____________________________________
% Require a function "delay_vec.m" to calculate the number of time-delayed vectors using a Takens' embedding theorem
% _____________ Reference __________
% 1) Kennel, M.B., Brown, R., and Abarbanel, H.D., 1992. Determining embedding dimension for phase-space reconstruction using a geometrical construction. Physical review A, 45(6), p.3403.
% 2) Abarbanel, H.D., Brown, R., Sidorowich, J.J., and Tsimring, L.S., 1993.  The analysis of observed chaotic data in physical systems. Reviews of modern physics, 65(4), p.1331.
% 3) Nayfeh, A.H., and Balachandran, B., 2008. Applied nonlinear dynamics: analytical,  computational and experimental methods. John Wiley & Sons.
%  Example: [FNN]=ARFM_fnn(x,10000,10,3,10,2)
%%_________________________________
% Subtract mean from the signal so that the signal will fluctuate around a zero line
% x = x - mean(x);
N=length(x);    % Get length of the signal 
% Generate a time vector from the known values of sampling frequency and length of the given signal
delta_t = 1/Fs;
% t = 1:N;
% t = t'*delta_t;

% Nmax = 10000;
% if (N>Nmax) %&& (strcmp(truncate,'on'))
%     disp(['Reducing number of datapoints from ',...
%         num2str(N),' to ',num2str(Nmax),' in order to run quickly'])
%     N=Nmax; x=x(end-N+1:end); t=t(end-N+1:end);
% end

% The algorithm to detect false nearest neighbors is based on the steps suggested by Kennel et al. (1992)
 
for k=1:dmax        % Loop to calculate false neighbors for various values of embedding dimension
    N1=floor(N-k*tao);     % Calculating maximum number of delay vectors for given dimension 		(d), signal length (N), and time delay (tao); i.e., [N-(d-1)*tao]
    Y=ARFM_delay_vec(x,tao,k,N1);    % Find number of delayed vectors of a signal from the given values of embedding dimension and delay
    FNN(k,1)=0;     % Initializing the value
    for j=1:N1      % Loop to calculate false neighbors of a phase space trajectory at a given embedding dimension
        Y0=ones(N1,1);  % Generate a vector of ones.
        Y1=Y0*Y(j,:);   % Generate the copies of a given vector equal to total number of delayed vectors
        R=sqrt(sum((Y-Y1).^2,2));   % Calculate the distance of a given point of the phase space trajectory with all other points
        [a, b]=sort(R);         % Sort the distances in ascending order, it is necessary to calculate the nearest neighbor    
		% Pick second minima of the distances as a first nearest neighbor, as the first minima is always zero - a distance of the point with itself
        NN=b(2);    % Position of the nearest neighbor on the phase space trajectory
        ND=a(2);    % The distance of the nearest neighbor from the given point on the trajectory.......[ Eq. (2) in the paper by Kennel et al. (1992)]
        Rd=(x(j+k*tao)-x(NN+k*tao));    % Calculate the distance of the nearest neighbour with all points in the increase dimensional space
        Rd1=sqrt(Rd.^2+ND.^2);          % The distance of points on the trajectory due to increasing in dimension from d to d+1......[ Eq. (3) in the paper by Kennel et al. (1992)]
        
		% Condition to check the falseness of nearest neighbors 
        if abs(Rd)/ND > Rtol
            FNN(k,1)=FNN(k,1)+1;       % [ Eq. (4) in the paper by Kennel et al. (1992)]
        elseif Rd1/std(x) > Atol
            FNN(k,1)=FNN(k,1)+1;       % [ Eq. (5) in the paper by Kennel et al. (1992)]
        end
    end
end
		% Calculate the percentage of false nearest neighbors for every embedding dimension
FNN=(FNN./FNN(1,1))*100;
