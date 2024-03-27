%  程序名称: CS-MSGM(1,N)模型参数估计及参数优化
%          基于布谷鸟算法的混频数据灰色模型，利用影响因素的高频数据序列预测系统行为因素低频数据序列
%  数据输入: 混频数据
%  结果输出: 模型参数; 混频滞后权重参数; 低频数据预测值
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
hf_data3 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","J2:J243"); % 读入table
hf_data3 = hf_data3{:,:}'; % table 转 matrix  % 行数据
% 自变量高频数据2（季度）
hf_data2 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","G2:G89"); % 读入table
hf_data2 = hf_data2{:,:}'; % table 转 matrix  % 行数据
% 自变量低频数据3（月度）
hf_data1 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","D2:D23"); % 读入table
hf_data1 = hf_data1{:,:}'; % table 转 matrix  % 行数据


%%%%% 时间序列长度 %%%%% 
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3);
U_len = [U1_len,U2_len,U3_len];
m = U_len./X_len;     % 高频因素与因变量的频率倍差

%%%%% 模型参数输入 %%%%% 
object = 3;                                       % 因素指标个数
test_len = 5;                                     % 测试集长度
train_len = X_len-test_len;                      % 训练集长度
lambda = [1,7,21];

% For reproducibility, set the random seed.
rng(0); % rng default

%% %%%%%%%%%%%%%%% 利用布谷鸟算法选择最佳的多项式参数lambda、theta %%%%%%%%%%%%%%%%%%%
% Change this if you want to get better results
    % Number of nests (or different solutions)
    n = 30;%鸟巢数量
    % Discovery rate of alien eggs/solutions
    pa = 0.25;% Probability of building a new nest(After host bird find exotic bird eggs)
    N_IterTotal = 300;%迭代次数
    %% Simple bounds of the search domain
    nd = 6; %问题的维度,一个鸟巢鸟蛋的个数 Dimensionality of solution
    % Lower bounds
    Lb = -5 * ones(1,nd); %函数下界
    % Upper bounds
    Ub = 5 * ones(1,nd);%函数上界

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

%% %%%%%% 这是根据最优值计算得到的拟合误差和滞后 %%%%%% 
[par,LF_simdata,APE,MAPE_train,MAPE_test,ENet] = fx_MSGM14(lf_data,hf_data1,hf_data2,hf_data3,...
    train_len,lambda,bestnest);     % 带入GWO寻优结果 bm_theta = bestnest
Fiting_value = LF_simdata';
Result = [lf_data',LF_simdata',abs(APE)'];

%% 真实值与预测值误差比较
figure
plot(lf_data,'bo-','linewidth',1.2)
hold on
plot(LF_simdata,'r*-','linewidth',1.2)
legend('真实值','模拟值')
xlabel('测试样本编号'),ylabel('指标值')
title('MSGM_{CS}真实值与模拟值的对比')
set(gca,'fontsize',12)

%计算误差
error = LF_simdata-lf_data;      %预测值和真实值的误差
APE = (error./lf_data)*100;
MAPE_train = mean(abs(APE(1:train_len)));
MAPE_test = mean(abs(APE(train_len+1:end)));
MAPE = [MAPE_train,MAPE_test];
RMSE_test = sqrt(mean(error(train_len+1:end)*error(train_len+1:end)'));
R2 = 1-sum(error*error')/sum((lf_data-mean(lf_data)).^2);
Theil_U = sqrt(sum(error*error'))/sqrt(sum(lf_data.^2));
r = corrcoef(LF_simdata,lf_data);    %corrcoef计算相关系数矩阵，包括自相关和互相关系数

%%%%%% 结果输出
Result = [lf_data', LF_simdata', abs(APE)'];
disp(['num         Actual_value     Fiting_value          APE']);
for i=1:X_len
    disp([i,lf_data(i),LF_simdata(i),abs(APE(i))])
end
MAPE = [MAPE_train,MAPE_test,RMSE_test,R2,Theil_U];
disp('MAPE_train       MAPE_test       RMSE_test       R2        Theil_U');
disp(MAPE);


% %% %%%%%%%%% 滞后参数权重
% bm_theta = bestnest;
% logwei_expalmon1 = [];          % Expalmon多项式权重生成
% for i = 1:lambda(1)+1
%     logwei_expalmon1(i,1) = exp(bm_theta(1)*(i-1)+bm_theta(2)*(i-1)^2);
% end
% ExpAlmon1 = logwei_expalmon1/sum(logwei_expalmon1,'all');     % 列和为1
% 
% logwei_expalmon2 = [];          % Expalmon多项式权重生成
% for i = 1:lambda(2)+1
%     logwei_expalmon2(i,1) = exp(bm_theta(3)*(i-1)+bm_theta(4)*(i-1)^2);
% end
% ExpAlmon2 = logwei_expalmon2/sum(logwei_expalmon2,'all');     % 列和为1
% 
% logwei_expalmon3 = [];          % Expalmon多项式权重生成
% for i = 1:lambda(3)+1
%     logwei_expalmon3(i,1) = exp(bm_theta(5)*(i-1)+bm_theta(6)*(i-1)^2);
% end
% ExpAlmon3 = logwei_expalmon3/sum(logwei_expalmon3,'all');     % 列和为1


