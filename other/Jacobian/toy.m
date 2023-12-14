clear all; clc; close all

%% Load data and check dimensions

Nq = 6;
Nr = 10;

x = rand([Nr, 1]);
x = x/norm(x);
A = rand([Nq, Nr]);


%% ANALYTICAL JACOBIAN
y = F(x, A);

sins = -sin(x);
J = zeros(Nq,Nr);

for i = 1:Nq
    for j = 1:Nr
        J(i,j) = A(i,j) * -sin(x(j));
    end
end
% 


J = A .* -sin(x).';

%% NUMERICAL
figure(); hold on
for eps_i = 9:-.1:-9
% for eps = [1, 1e+3, 1e+9, 1e-12]
    eps = 10^eps_i;
    dbdy = zeros(Nq,Nr);


    for i = 1:Nr

        x_tilde = x;

        x_tilde(i) = x(i) + eps;
    

        dbdy(:,i) = (F(x_tilde, A) - F(x, A))/eps ;
    end
    
    % Check difference
    error1 = norm((J - dbdy))/norm(J);
    plot(eps, error1, 'bo')

%     disp(error1)

end
xlabel('epsilon [exponent]')
ylabel('error')
set(gca, 'YScale', 'log','XScale', 'log')


%% ===================================================================== %

function [y] = F(x, A)
    y = A*cos(x);
end

