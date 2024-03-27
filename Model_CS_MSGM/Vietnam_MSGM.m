%  程序功能: 混频时间序列绘图 
clc;clear all

%% %%%%%%%%%%%%%%%%%%%%%%%% read data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add path
addpath('/Users/anyimeng/Research/Datasets/Vietnam/Climate_daily') ; savepath
addpath('/Users/anyimeng/Research/Datasets/Vietnam/Fulldata') ; savepath

% daily data of climate
sheetNames = sheetnames("days30_HaNoi.xlsx");
data_dailyclimate_HaNoi = [];
for i = 2:length(sheetNames)
    HaNoi = readtable('days30_HaNoi.xlsx',"Sheet",sheetNames{i},...
        "ReadRowNames",false,"ReadVariableNames",true,'VariableNamingRule','preserve'); % 读入table
    HaNoi = HaNoi{:,2:5};
    data_dailyclimate_HaNoi = [data_dailyclimate_HaNoi; HaNoi];
end

% monthly data of cases
data_monthlycases_HaNoi = readtable('Hà Nội.xlsx', 'ReadVariableNames',false);
% monthly data of Dia. case in 01/2000-—08/2012 
data_monthlyDia_HaNoi = data_monthlycases_HaNoi{37:188,20};


%%  %%%%%%%%%%%%%%%%%%%%%%%% MSGM modeling %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
normadata_dailyclimate_HaNoi = normalize(data_dailyclimate_HaNoi, 'range'); % [a b] 形式的区间（默认为 [0 1]）
normadata_monthlyDia_HaNoi = normalize(data_monthlyDia_HaNoi,'range');
lf_data = normadata_monthlyDia_HaNoi(25:end,:)';

normadata_dailyclimate_HaNoi = normalize(data_dailyclimate_HaNoi, 'range'); % [a b] 形式的区间（默认为 [0 1]）
hf_data = normadata_dailyclimate_HaNoi(721:end,:)';

hf_data1 = hf_data(2,:);
hf_data2 = hf_data(3,:);
hf_data3 = hf_data(4,:);

%%%%% 时间序列长度 %%%%% 
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3);
U_len = [U1_len,U2_len,U3_len];
m = U_len./X_len;     % 高频因素与因变量的频率倍差

%%%%% 模型参数输入 %%%%% 
object = 3;                                       % 因素指标个数
test_len = 25;                                     % 测试集长度
train_len = X_len-test_len;                      % 训练集长度
lambda = 2*m-1;

% For reproducibility, set the random seed.
rng(0); % rng default

%% %%%%%%%%%%%%%%% 利用布谷鸟算法选择最佳的多项式参数lambda、theta %%%%%%%%%%%%%%%%%%%
% Change this if you want to get better results
    % Number of nests (or different solutions)
    n = 50;%鸟巢数量
    % Discovery rate of alien eggs/solutions
    pa = 0.25;% Probability of building a new nest(After host bird find exotic bird eggs)
    N_IterTotal = 500;%迭代次数
    %% Simple bounds of the search domain
    nd = 6; %问题的维度,一个鸟巢鸟蛋的个数 Dimensionality of solution
    % Lower bounds
    Lb = -25 * ones(1,nd); %函数下界
    % Upper bounds
    Ub = 25 * ones(1,nd);%函数上界

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



