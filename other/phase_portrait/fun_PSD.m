function [f,PSD] = fun_PSD(dt,x)
% Function that computes the Power Spectral Density.
% - Inputs: 
%       - dt: sampling time 
%       - x: signal to compute the PSD
% - Outputs: 
%       - f: corresponding frequencies
%       - PSD: Power Spectral Density
% =========================================================================

% 
%     Fs             =    1/dt;              % fundamental frequency  
%     m              =    numel(x);
%     n              =    pow2(nextpow2(m)); % or = m 
%     % Fast Fourier transform the signal
%     y              =    fft(x,n);
%     % Power spectral density
%     p2             =    y.*conj(y)/n;
%     PSD            =    p2(1:n/2+1);
%     % Frequencies
%     f              =    (0:n/2)*(Fs/n);
        
    
    %% FRANCISCO'S APPROACH
    len = numel(x);
    f = linspace(0.0, 1.0/(2.0*dt), len/2);
    yt = fft(x);
    PSD = 2.0/len * abs(yt(1:floor(len/2)));

    
end
 
 
 