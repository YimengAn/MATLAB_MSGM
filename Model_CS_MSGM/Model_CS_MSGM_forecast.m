%  程序名称: CS-MSGM(1,N)_forecast
%          基于布谷鸟算法的混频数据灰色模型，利用影响因素的高频数据序列预测系统行为因素低频数据序列
%  数据输入: Training_set; High-frequency data of independent variables;
%  结果输出: MSGM parameters; 
%          out-of-sample predicted value of low-frequency variable;
%  注意: 函数文件 fx_MSGM1N.m、initialization.m 与本程序需在同一目录下

clc,clear all;
format short g;
fprintf(2,'布谷鸟算法参数优化的混频数据灰色多变量预测模型CS-MSGM(1,N)：\n')

%% R2020b_MATLAB 读入数据
%% %%%%%%%%%%%%%%%%%%%%%%%% 将.xls数据文件添加到路径 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mkdir('/Users/anyimeng/科研/MATLABCode/MATLAB_Mix.data/Table') ;  
% addpath('/Users/anyimeng/科研/MATLABCode/MATLAB_Mix.data/Table') ; 
% savepath /Users/anyimeng/科研/MATLABCode/MATLAB_Mix.data/Table/pathdef.m ;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 数据输入 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 混频数据时间序列
sheetNames = sheetnames("China_3E.xlsx");
% 因变量低频数据
lf_data = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","C2:C23"); % 读入table
lf_data = lf_data{:,:}';  % table 转 matrix  % 行数据
% 自变量高频数据1（年度）
hf_data1 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","D2:D23"); % 读入table
hf_data1 = hf_data1{:,:}'; % table 转 matrix  % 行数据
% 自变量高频数据2（季度）
hf_data2 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","G2:G89"); % 读入table
hf_data2 = hf_data2{:,:}'; % table 转 matrix  % 行数据
% 自变量低频数据3（月度）
hf_data3 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","J2:J243"); % 读入table
hf_data3 = hf_data3{:,:}'; % table 转 matrix  % 行数据

%%%%% 时间序列长度 %%%%% 
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3);
U_len = [U1_len,U2_len,U3_len];
m = U_len./X_len;     % 高频因素与因变量的频率倍差

%%%%% 模型参数输入 %%%%% 
object = 3;                                       % 因素指标个数
test_len = 0;                                     % 测试集长度
train_len = X_len-test_len;                       % 训练集长度
lambda = [1,7,21];


%% For reproducibility, set the random seed.
rng(0); % rng default

%% %%%%%%%%%%%%%%% 利用布谷鸟算法选择最佳的多项式参数lambda、theta %%%%%%%%%%%%%%%%%%%
% Change this if you want to get better results
    % Number of nests (or different solutions)
    n=30;%鸟巢数量
    % Discovery rate of alien eggs/solutions
    pa=0.25;% Probability of building a new nest(After host bird find exotic bird eggs)
    N_IterTotal=300;%迭代次数
    %% Simple bounds of the search domain
    % Lower bounds
    nd=6; %问题的维度,一个鸟巢鸟蛋的个数 Dimensionality of solution
    % Low bounds
    Lb=-5*ones(1,nd); %函数下界
    % Upper bounds
    Ub=5*ones(1,nd);%函数上界
    
    % Random initial solutions
    for i=1:n
        nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb)); %初始化寄主的鸟巢
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
        %通过levy飞行产生一个解Generate new solutions by Levy flights
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
            fmin=fnew;%%%%%%%最优适应度值
            bestnest=best;%%%%%%%%待求参数
        end
    end %% End of iterations
%     Monte_Carlo_CS(kk)=fmin;
%% Post-optimization processing
%% Display all the nests
disp(strcat('Total number of iterations=',num2str(N_iter)));
fmin   %%%%这是适应度值，拟合部分的MAPE
bestnest %%%%%%这是待求的参数




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 外推预测 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[par,LF_simdata,APE,MAPE_train,MAPE_test,ENet] = ...
    fx_MSGM14(lf_data,hf_data1,hf_data2,hf_data3,train_len,lambda,bestnest);

% %%%%%%%%%% 输入建模数据 %%%%%%%%%%
% 样本外预测步数
outsample_len = 5;  

% 自变量高频数据1（年度）
hf_data1_out = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China_forecast',"Range","D18:G22"); % 读入table
hf_data1_out = hf_data1_out{:,:}'; % table 转 matrix  % 行数据
% 自变量高频数据2（季度）
hf_data2_out = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China_forecast',"Range","J18:M37"); % 读入table
hf_data2_out = hf_data2_out{:,:}'; % table 转 matrix  % 行数据
% 自变量低频数据3（月度）
hf_data3_out = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China_forecast',"Range","P18:S72"); % 读入table
hf_data3_out = hf_data3_out{:,:}'; % table 转 matrix  % 行数据


%% out-of-sample forecast
Scio = 4;                                         % Scio 场景假设
LF_data = [lf_data,zeros(1,outsample_len)];
HF_data1 = [hf_data1,hf_data1_out(Scio,:)];
HF_data2 = [hf_data2,hf_data2_out(Scio,:)];
HF_data3 = [hf_data3,hf_data3_out(Scio,:)];

% %%%%%%%%%% 模拟生成 %%%%%%%%%%
[LF_fitting,APE,MAPE_train] = fx_MSGM14_forecast...
    (par,LF_data,HF_data1,HF_data2,HF_data3,outsample_len,lambda,bestnest)

%% %%%%%%%%%%%%%%%%%%%%%%%%真实值与预测值误差比较%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
plot(lf_data,'bo-','linewidth',1.2)
hold on
plot(LF_fitting,'r*-','linewidth',1.2)
% legend('期望值','预测值')
xlabel('测试样本编号'),ylabel('指标值')
title('BP测试集预测值和期望值的对比')
set(gca,'fontsize',12)