clear all; clc

%% Load data and check dimensions
load("ESN_data.mat")
b = b.';
r = r.';

g = norm.';
Nq = length(b);
Nr = length(r);

clear norm

% Force Win to be Nr x Nq
if size(Win, 2) > size(Win, 1)
    Win = Win.';
end
% Force Wout to be Nq x Nr
if size(Wout, 1) > size(Wout, 2)
    Wout = Wout.';
end

%% ANALYTICAL JACOBIAN
Win_1 = Win(:, 1:end-1);
Wout_1 = Wout(:, 1:end-1);

[bout, rout, WCout] = step(b, r, Win, W, Wout, sigma_in, rho, g);
% bout = [-1913.49359458, -6399.47636438, -6400.88752921, -3710.01686313,
%        -2423.6175931 , -1491.14510932] <---- true result from ESN code



J = zeros(Nq,Nq);

% Option (i) if rin is not a function of bin. "Open-loop Jacobian"
% for j = 1:Nq
%     for k = 1:Nr
%         J(:,j) = J(:,j) + Wout_1(:,k) * (1-rout(k)^2) * ...
%             (sigma_in * Win_1(k,j) / g(j));
%     end
% end

tic
% Option (ii) if rin is a function of bin. "Closed-loop Jacobian"
for j = 1:Nq
    for k = 1:Nr
        J(:,j) = J(:,j) + Wout_1(:,k) * (1-rout(k)^2) * ...
            (sigma_in * Win_1(k,j) / g(j) + rho * WCout(k,j));
    end
end
toc


tic    
J2 = Wout_1 * ((1 - rout.^2) .* (sigma_in * Win_1./g.' + rho * WCout));
toc

disp(norm(J-J2)) % Check matrix formulaiton


%% NUMERICAL
figure()
hold on

for eps_i = -5:0.05:3
    eps = 10^eps_i;
    dbdy = zeros(Nq,Nq);
    for i = 1:Nq
        b_tilde = b;
        b_tilde(i) = b_tilde(i) + eps;
    
        [bout_tilde, ~] = step(b_tilde, r, Win, W, Wout, sigma_in, rho, g);

        dbdy(:,i) = (bout_tilde - bout)/(eps) ;
    end
    
    % Check difference
    error1 = norm(J - dbdy)/norm(J);
    plot(eps, error1, 'bo')

    % Check difference
    error1 = norm(J2 - dbdy)/norm(J2);
    plot(eps, error1, 'rx')

end
xlabel('epsilon')
ylabel('error')
set(gca, 'YScale', 'log','XScale', 'log')




%% ===================================================================== %

function [bout, rout, WCout] = step(b, r, Win, W, Wout, sig, rho, g)
    Win_1 = Win(:, 1:end-1);
    Win_2 = Win(:, end);
    Wout_1 = Wout(:, 1:end-1);
    Wout_2 = Wout(:, end);

%     % 1) Inverse as column space
%     WW = (Wout_1.'*Wout_1)+1E12*eye(100); % Need large pert to stabilese
%     Cout = WW^-1 * Wout_1.';


    % 2) Find WCout = W * Wout_1^-1 by solviong the problem as
    %    WCout * Wout_1 = W => Wout_1^T * WCout^T = W^T
    WCout = mldivide(Wout_1.', W.').';

%     % Option (i). Open-loop
%     rout = tanh(sig * Win_1 * (b./g) + 0.1 * sig * Win_2 + rho * W * r);

    % Option (ii). Closed-loop
    rout = tanh(sig * Win_1 * (b./g) + 0.1 * sig * Win_2 + rho * WCout*b);

    bout = Wout_1 * rout + Wout_2;
end

