function [v, L] = ARFM_ami(x, lag)
% This code will help in finding the optimum time lag required for the 
% construction of phase portrait from a discrete time series.
% _____________ Inputs ______________________
% x:        The 1D time series
% lag:      Number of lags
% _____________ Output ______________________
% v:        Average mutual information (AMI)
% _____________ Reference ___________________
% Fraser, A.M. and Swinney, H.L., 1986. Independent coordinates for 
% strange attractors   from mutual information. 
% Physical review A, 33(2), p.1134.
%____________________________________________
% This concept is based on information theory.
% AMI suggests how much information we can obtain from the 
% measurement of a(i), chosen from set A, about the measurement 
% of b(i), chosen from set B.
% The first local minima of AMI is chosen as the optimum time delay
% (Fraser & Swinney 1986).
%%---------------------------------------------------------
y = x;      % created new vector same as given data set
x = x(:);   % transform original vector from raw to column
y = y(:);   % transform new vector from raw to column
N = length(x);  % calculate number of data points present in the signal
 
% Specify the value of time lag (number of samples in integer) up to 
% which the mutual information is needed to estimate
L = 0:lag;
% Normalize x and y to lie between 0 and 1 <-- only do if x not 0
x = x - min(x);
x = x*(1-eps)/max(x);
y = y - min(y);
y = y*(1-eps)/max(y); 

v = zeros(size(L));     % Initialize the AMI vector

for i = 1:length(L)
    % Define the number of bins
    % Split the sample interval in log2 spaced bins 
    % The logarithm of basis 2 makes the measurement of information in bits.
    bins = ceil(log2(N - L(i)));
    % Distributing data in the bin in which it is located. 
    % The numbering goes from 1 to number of bins, i.e., 1, 2, 3, ..., so on
    binx = floor(x*bins) + 1;
    biny = floor(y*bins) + 1;
    % Initialize the joint probability density vector
    Pxy = zeros(bins);
    % Find joint probability Pxy for different lags
    for j = 1:((N - L(i)))
        k = floor(j + L(i));
        Pxy(binx(k), biny(j)) = Pxy(binx(k), biny(j)) + 1;
    end
    % Normalize Pxy
    Pxy = Pxy/(N - L(i));
    % In order to avoid probability value tending to infinity (lag(0)), a floating point  number is added
    Pxy = Pxy + eps;
    % Calculating the marginal probability of x and y vectors
    % Column sum for marginal probability of x
    Px = sum(Pxy,2);
    % Row sum for marginal probability of y
    Py = sum(Pxy,1);
    % Calculating Average Mutual Information of a given time series at every time lag
    q = Pxy./(Px*Py);
    AMI = Pxy.*log2(q);
    v(i) = sum(AMI(:));     % [Eq. 9 in Fraser & Swinney, 1986]
    %     Shannon entropy can be calculated from following equations    
    %    [Eq. 2 in Fraser & Swinney, 1986]
    %     Hx = -sum(Px.*log2(Px));
    %     Hy = -sum(Py.*log2(Py));
    %     Hxy = -sum(Pxy(:).*log2(Pxy(:)));
    %     v(i) = Hx+Hy-Hxy;
end
end
