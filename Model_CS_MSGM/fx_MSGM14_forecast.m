function [LF_fitting,APE,MAPE_train] = fx_MSGM14_forecast...
    (par,LF_data,HF_data1,HF_data2,HF_data3,outsample_len,lambda,bm_theta)
%UNTITLED 高频影响因素MSGM(1,4)模型
%   LF_data = 低频时间序列（年度）                             % 低频因变量行数据
%   HF_data1 = 第一个自变量时间序列（月度）                      % 高频自变量行数据
%   HF_data2 = 第二个自变量时间序列（季度）                      % 高频自变量行数据
%   HF_data3 = 第三个自变量时间序列（年度）                      % 低频自变量行数据
%       建模使用的 LF_data 和 HF_data 数据，均为预处理后的数据
%   outsample_len = 样本外预测步长
%   lambda = [lambda1,lambda2,lambda3]                      % 自变量对因变量的最大滞后期
%   bm_theta = [theta11,theta12,theta21,theta22,theta31,theta32]      % 自变量高频数据赋权参数  
%   par = [alpha0,alpha1,beta1,beta2,beta3,eta]             % beta个数 = object
%   LF_simdata = 低频数据模拟生成序列



%% %%%%%%%%%% 输入建模数据 %%%%%%%%%%
% 自变量个数
object = 3;
% 数据样本
x_len = length(LF_data)                         
u1_len = length(HF_data1);
u2_len = length(HF_data2);  
u3_len = length(HF_data3); 
m = [u1_len/x_len,u2_len/x_len,u3_len/x_len];     % 高频因素与因变量的频率倍差
T = [1:1:x_len];


%% %%%%%%%%%% 高频数据多项式权重 %%%%%%%%%%
% bm_theta = Beta_pos;
logwei_expalmon1 = [];          % Expalmon多项式权重生成
for i = 1:lambda(1)+1
    logwei_expalmon1(i,1) = exp(bm_theta(1)*(i-1)+bm_theta(2)*(i-1)^2);
end
ExpAlmon1 = logwei_expalmon1/sum(logwei_expalmon1,'all');     % 列和为1

logwei_expalmon2 = [];          % Expalmon多项式权重生成
for i = 1:lambda(2)+1
    logwei_expalmon2(i,1) = exp(bm_theta(3)*(i-1)+bm_theta(4)*(i-1)^2);
end
ExpAlmon2 = logwei_expalmon2/sum(logwei_expalmon2,'all');     % 列和为1

logwei_expalmon3 = [];          % Expalmon多项式权重生成
for i = 1:lambda(3)+1
    logwei_expalmon3(i,1) = exp(bm_theta(5)*(i-1)+bm_theta(6)*(i-1)^2);
end
ExpAlmon3 = logwei_expalmon3/sum(logwei_expalmon3,'all');     % 列和为1


%% %%%%%%%%%% 模型参数估计 %%%%%%%%%%
V1=[];
for i = 2:x_len
    u1_ti = HF_data1(1,(m(1)*i-lambda(1)):m(1)*i);
    V1(i) = u1_ti*ExpAlmon1;
end

V2=[];
for i = 2:x_len
    u2_ti = HF_data2(1,(m(2)*i-lambda(2)):m(2)*i);
    V2(i) = u2_ti*ExpAlmon2;
end

V3=[];
for i = 2:x_len
    u3_ti = HF_data3(1,(m(3)*i-lambda(3)):m(3)*i);
    V3(i) = u3_ti*ExpAlmon3;
end

V = [V1;V2;V3];
Vt = V-V(:,1);

%% %%%%%%%%%% 模拟生成 %%%%%%%%%%
% 时间响应式
a0 = par(1)/(1-0.5*par(2));
a1 = par(2)/(1-0.5*par(2));
pb = par(3:end-1);                                     % theta
bmB = pb/(1-0.5*par(2));
eta = par(end);
LF_simdata(1) = LF_data(1); 
for k = 2:x_len;
    % 累加生成序列
    LF_cum = cumsum(LF_simdata);                         % Lf_data的累加生成序列
    ZLF = 0.5*LF_cum(1:end-1)+0.5*LF_cum(2:end);         % LF_cum的均值生成序列   
    LF_simdata(k) = a0*(k-1)+a1*LF_cum(k-1)+bmB'*Vt(:,k)+eta;
end
LF_fitting = LF_simdata'

%%  %%%%%%%%%% 模拟和预测精度 %%%%%%%%%%
APE = ((LF_simdata(1:x_len-outsample_len)-LF_data(1:x_len-outsample_len))./LF_data(1:x_len-outsample_len))*100;
MAPE_train = mean(abs(APE));

end
