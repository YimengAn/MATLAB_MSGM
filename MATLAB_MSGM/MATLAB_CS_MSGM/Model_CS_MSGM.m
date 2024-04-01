% Program Name: Parameter Estimation and Optimization of CS-MSGM(1,N) Model
%              Gray Multivariable Prediction Model based on Cuckoo Search Algorithm,
%              using High-frequency Data Series of Influencing Factors to Predict Low-frequency Data Series
% Data Input: Mixed-frequency sampling Data
% Result Output: Model Parameters; High-frequency Lag Weight Parameters; Low-frequency Data Predicted Values
% Note: Function files fx_MSGM1N.m, initialization.m, and this program should be in the same directory

clc;
clear all;
format short g;
fprintf(2, 'CS_MSGM(1,N) model:\n')

%% R2020b_MATLAB Data Input
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data Input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mixed-frequency data time series
sheetNames = sheetnames("RawData.xlsx");
% Dependent variable low-frequency data
lf_data = readtable('RawData.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","D3:D24"); 
lf_data = lf_data{:,:}'; 
% Independent variable low-frequency data (annual)
hf_data1 = readtable('RawData.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","H3:H24"); 
hf_data1 = hf_data1{:,:}'; 
% Independent variable high-frequency data (quarterly)
hf_data2 = readtable('RawData.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","L3:L90"); 
hf_data2 = hf_data2{:,:}'; 
% Independent variable high-frequency data (monthly)£©
hf_data3 = readtable('RawData.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","P3:P244"); 
hf_data3 = hf_data3{:,:}';
%%%%% Time Series Length %%%%%
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3);
U_len = [U1_len,U2_len,U3_len];
m = U_len./X_len;    % Frequency ratio between hf data and lf data

%%%%% Model Parameter Input %%%%%
object = 3;                                       % Number of factor indices
test_len = 5;                                     % Test set length
train_len = X_len - test_len;                     % Training set length
lambda = [1, 7, 21];

% For reproducibility, set the random seed.
rng(0); % rng default

%% %%%%%%%%%%%%%%%%%%%%%%%% Cuckoo Search Algorithm %%%%%%%%%%%%%%%%%%%%%%%%%
% Change this if you want to get better results
    % Number of nests (or different solutions)
    n = 30;
    % Discovery rate of alien eggs/solutions
    pa = 0.25;% Probability of building a new nest(After host bird find exotic bird eggs)
    N_IterTotal = 300;  % Number of iterations
    %% Simple bounds of the search domain
    nd = 6; % Dimensionality of solution
    % Lower bounds
    Lb = -5 * ones(1,nd); 
    % Upper bounds
    Ub = 5 * ones(1,nd);

    % Random initial solutions
    for i=1:n
        nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb)); 
    end
    % Get the current best
    fitness=10^10*ones(n,1);
    [fmin,bestnest,nest,fitness]=get_best_nest(lf_data,hf_data1,hf_data2,hf_data3,...
        train_len,lambda,nest,nest,fitness);
    
    [fmin,K]=min(fitness) ;
    best=nest(K,:);
    N_iter=0;
    % Levy flights
    n=size(nest,1);
    % Levy exponent and coefficient
    % For details, see equation (2.21), Page 16 (chapter 2) of the book
    % X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).
    
    beta=3/2;
    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    
    for j=1:n
        s=nest(j,:);
        % This is a simple way of implementing Levy flights
        % For standard random walks, use step=1;
        %% Levy flights by Mantegna's algorithm
        % Generate new solutions by Levy flights
        u=randn(size(s))*sigma;
        v=randn(size(s));
        step=u./abs(v).^(1/beta);
        
        % In the next equation, the difference factor (s-best) means that
        % when the solution is the best solution, it remains unchanged.
        stepsize=0.01*step.*(s-best);
        % Here the factor 0.01 comes from the fact that L/100 should the typical
        % step size of walks/flights where L is the typical lenghtscale;
        % otherwise, Levy flights may become too aggresive/efficient,
        % which makes new solutions (even) jump out side of the design domain
        % (and thus wasting evaluations).
        % Now the actual random walks or flights
        s=s+stepsize.*randn(size(s));
        % Apply simple bounds/limits
        nest(j,:)=simplebounds(s,Lb,Ub);
    end
    %% Starting iterations
    for iter=1:N_IterTotal
        % Generate new solutions (but keep the current best)
        %      new_nest=get_cuckoos(nest,bestnest,Lb,Ub);
        new_nest=nest;
        [fnew,best,nest,fitness]=get_best_nest(lf_data,hf_data1,hf_data2,hf_data3,...
            train_len,lambda,nest,new_nest,fitness);
       
        % Update the counter
        N_iter=N_iter+n;
        % Discovery and randomization
        new_nest=empty_nests(nest,Lb,Ub,pa) ;
        
        % Evaluate this set of solutions
        [fnew,best,nest,fitness]=get_best_nest(lf_data,hf_data1,hf_data2,hf_data3,...
            train_len,lambda,nest,new_nest,fitness);
        
        % Update the counter again
        N_iter=N_iter+n;
        % Find the best objective so far
        if fnew<fmin
            fmin=fnew;        
            bestnest=best;   
        end
    end %% End of iterations

disp(strcat('Total number of iterations=', num2str(N_iter)));
fmin       % Best fitness value
bestnest   % Best parameters


%% %%%%%% Calculate Fit Errors and Lag based on Optimal Values %%%%%% 
[par, LF_simdata, APE, MAPE_train, MAPE_test, ENet] = fx_MSGM14(lf_data, hf_data1, hf_data2, hf_data3, ...
    train_len, lambda, bestnest);
Fitting_value = LF_simdata';
Result = [lf_data', LF_simdata', abs(APE)'];

%% Compare Actual Values with Predicted Values
figure
plot(lf_data, 'bo-', 'linewidth', 1.2)
hold on
plot(LF_simdata, 'r*-', 'linewidth', 1.2)
legend('Actual Values', 'Simulated Values')
xlabel('Test Sample Number'), ylabel('Indicator Value')
title('Comparison of Actual and Simulated Values for MSGM_{CS}')
set(gca, 'fontsize', 12)

% Calculate errors
error = LF_simdata - lf_data;      
APE = (error ./ lf_data) * 100;
MAPE_train = mean(abs(APE(1:train_len)));
MAPE_test = mean(abs(APE(train_len+1:end)));
MAPE = [MAPE_train, MAPE_test];
RMSE_test = sqrt(mean(error(train_len+1:end) * error(train_len+1:end)'));
R2 = 1 - sum(error * error') / sum((lf_data - mean(lf_data)).^2);
Theil_U = sqrt(sum(error * error')) / sqrt(sum(lf_data.^2));
r = corrcoef(LF_simdata, lf_data);    

%%%%% Result Output
Result = [lf_data', LF_simdata', abs(APE)'];
disp(['num         Actual_value     Fiting_value          APE']);
for i = 1:X_len
    disp([i, lf_data(i), LF_simdata(i), abs(APE(i))])
end
MAPE = [MAPE_train, MAPE_test, RMSE_test, R2, Theil_U];
disp('MAPE_train       MAPE_test       RMSE_test       R2        Theil_U');
disp(MAPE);
