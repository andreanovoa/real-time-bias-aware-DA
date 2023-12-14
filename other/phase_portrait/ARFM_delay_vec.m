function [Y]=ARFM_delay_vec(x,tao,d,N1)
% This code constructs the delay vectors from a give scalar time series using appropriate value of embedding 
% dimension and the delay time. It is based on the theorem of time delay embedding proposed by Takens (1981)
 
% _____________ Inputs _________________
%  x:		 	A scalar time series
%  tao: 		The delay time (calculated from AMI)
%  d: 		The embedding dimension (calculated either from FNN method or Cao's method)
%  N1:		Number of delayed vectors (N1=length(x)-(d1-)*tao)
% _____________ Outputs _______________ 
%  Y: A matrix of dimension [N1,d], where N is the number of delayed vectors calculated from the signal 
% and d is the dimension in which the signal is embedded
%%__________________________________
% Initializing a delay vector matrix, where 'N1' is the number of delay vectors and 'd' is the embedding dimension
Y=zeros(N1,d);   
for i=1:d   
    Y(:,d-i+1) = x(1+(i-1)*tao:N1+(i-1)*tao);  % Delay vectors are creased by lagging the signal with a given delay time  
end
