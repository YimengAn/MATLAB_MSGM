function [par,LF_simdata,APE,MAPE_train,MAPE_test,ENet] = fx_MSGM14(lf_data,hf_data1,hf_data2,hf_data3,train_len,lambda,bm_theta)
% fx_MSGM(1,M): Mixed-sampling frequency grey system model with M variables
% Inputs:
%   - lf_data: Low-frequency time series (annual)              % Row data of low-frequency dependent variable
%   - hf_data1: First independent variable time series (monthly)    % Row data of high-frequency independent variable
%   - hf_data2: Second independent variable time series (quarterly) % Row data of high-frequency independent variable
%   - hf_data3: Third independent variable time series (annual)    % Row data of low-frequency independent variable
%       lf_data and hf_data used for modeling are pre-processed data
%   - train_len: Length of the training set
%   - lambda: [lambda1, lambda2, lambda3]                   % Maximum lag periods of independent variables to dependent variable
%   - bm_theta: [theta11, theta12, theta21, theta22, theta31, theta32] % Weighting parameters for high-frequency data
% Outputs:
%   - par: [alpha0, alpha1, beta1, beta2, beta3, eta]      % Number of beta parameters = object
%   - LF_simdata: Simulated low-frequency data


%% Input raw data and modeling parameters %%
% Number of independent variables
object = 3;                                       
% Data samples
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3); 
m = [U1_len/X_len,U2_len/X_len,U3_len/X_len];    
    % Frequency ratio between independent variables and dependent variable

%% Input modeling data %%
% Training set for model parameters
LowF_data = lf_data(1:train_len);
HighF_data1 = hf_data1(1:train_len*m(1));
HighF_data2 = hf_data2(1:train_len*m(2));
HighF_data3 = hf_data3(1:train_len*m(3));
% Length of training set data
x_len = length(LowF_data);
u1_len = length(HighF_data1);
u2_len = length(HighF_data2);
u3_len = length(HighF_data3);
T = [1:1:x_len];
% Cumulative generation sequence
LF_cum = cumsum(lf_data);                         % Cumulative generation sequence of Lf_data
ZLF = 0.5*LF_cum(1:end-1)+0.5*LF_cum(2:end);      % Mean generation sequence of LF_cum


%% Polynomial weighting for high-frequency data %%
% bm_theta = Beta_pos;
logwei_expalmon1 = [];          % Expalmon polynomial
for i = 1:lambda(1)+1
    logwei_expalmon1(i,1) = exp(bm_theta(1)*(i-1)+bm_theta(2)*(i-1)^2);
end
ExpAlmon1 = logwei_expalmon1/sum(logwei_expalmon1,'all');     % Sum of columns is 1

logwei_expalmon2 = [];          % Expalmon polynomial
for i = 1:lambda(2)+1
    logwei_expalmon2(i,1) = exp(bm_theta(3)*(i-1)+bm_theta(4)*(i-1)^2);
end
ExpAlmon2 = logwei_expalmon2/sum(logwei_expalmon2,'all');     % Sum of columns is 1

logwei_expalmon3 = [];          % Expalmon polynomial
for i = 1:lambda(3)+1
    logwei_expalmon3(i,1) = exp(bm_theta(5)*(i-1)+bm_theta(6)*(i-1)^2);
end
ExpAlmon3 = logwei_expalmon3/sum(logwei_expalmon3,'all');     % Sum of columns is 1


%% Model parameter estimation %%
V1 = arrayfun(@(i) (hf_data1(1, (m(1) * i - lambda(1)):m(1) * i) * ExpAlmon1), 2:X_len);
V2 = arrayfun(@(i) (hf_data2(1, (m(2) * i - lambda(2)):m(2) * i) * ExpAlmon2), 2:X_len);
V3 = arrayfun(@(i) (hf_data3(1, (m(3) * i - lambda(3)):m(3) * i) * ExpAlmon3), 2:X_len);
V = [V1; V2; V3];

% Parameter estimation
X = LowF_data(2:end)';
time = [2:x_len]'-T(1);
one = ones(x_len-1,1);
bm_ZLF = ZLF(1:(train_len-1))';
bm_V = V(:,1:train_len-1)';
Xi = [time,bm_ZLF,bm_V,one];
par = lsqminnorm(Xi'*Xi,Xi'*X);


%% Simulation %%
% Time responsive
a0 = par(1)/(1-0.5*par(2));
a1 = par(2)/(1-0.5*par(2));
pb = par(3:end-1);                                     % theta
bmB = pb/(1-0.5*par(2));
eta = par(end);
LF_simdata(1) = LowF_data(1); 
for k = 2:X_len
    LF_simdata(k) = a0*(k-1)+a1*LF_cum(k-1)+bmB'*V(:,k-1)+eta;
end


%% Model Accuracy %%
% Absolute Percentage Error (APE)
ape = ((LF_simdata(2:train_len)-lf_data(2:train_len))./lf_data(2:train_len))*100;
APE = ((LF_simdata(1:end)-lf_data(1:end))./lf_data(1:end))*100;
MAPE_train = mean(abs(ape));
MAPE_test = mean(abs(APE(train_len+1:end)));
% ENet Algorithm
ENet = 0.5*mean((LF_simdata(2:end)-lf_data(2:end)).^2)+0.1*norm(par,1);     

end
