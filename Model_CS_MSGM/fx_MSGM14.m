function [par,LF_simdata,APE,MAPE_train,MAPE_test,ENet] = fx_MSGM14(lf_data,hf_data1,hf_data2,hf_data3,train_len,lambda,bm_theta)
%UNTITLED 高频影响因素MSGM(1,4)模型
%   lf_data = 低频时间序列（年度）                             % 低频因变量行数据
%   hf_data1 = 第一个自变量时间序列（月度）                      % 高频自变量行数据
%   hf_data2 = 第二个自变量时间序列（季度）                      % 高频自变量行数据
%   hf_data3 = 第三个自变量时间序列（年度）                      % 低频自变量行数据
%       建模使用的 lf_data 和 hf_data 数据，均为预处理后的数据
%   train_len = 训练集长度
%   lambda = [lambda1,lambda2,lambda3]                      % 自变量对因变量的最大滞后期
%   bm_theta = [theta11,theta12,theta21,theta22,theta31,theta32]      % 自变量高频数据赋权参数  
%   par = [alpha0,alpha1,beta1,beta2,beta3,eta]             % beta个数 = object
%   LF_simdata = 低频数据模拟生成序列


%% %%%%%%%%%% 输入原始数据与建模参数 %%%%%%%%%%
% 自变量个数
object = 3;                                   
% train_len = 训练集长度
% 数据样本
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3); 
m = [U1_len/X_len,U2_len/X_len,U3_len/X_len];     % 高频因素与因变量的频率倍差


%% %%%%%%%%%% 输入建模数据 %%%%%%%%%%
% 模型参数训练集
LowF_data = lf_data(1:train_len);
HighF_data1 = hf_data1(1:train_len*m(1));
HighF_data2 = hf_data2(1:train_len*m(2));
HighF_data3 = hf_data3(1:train_len*m(3));
% 训练集数据长度
x_len = length(LowF_data);
u1_len = length(HighF_data1);
u2_len = length(HighF_data2);
u3_len = length(HighF_data3);
T = [1:1:x_len];
% 累加生成序列
LF_cum = cumsum(lf_data);                         % lf_data的累加生成序列
ZLF = 0.5*LF_cum(1:end-1)+0.5*LF_cum(2:end);      % LF_cum的均值生成序列


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
for i = 2:X_len
    u1_ti = hf_data1(1,(m(1)*i-lambda(1)):m(1)*i);
    V1(i) = u1_ti*ExpAlmon1;
end
V2=[];
for i = 2:X_len
    u2_ti = hf_data2(1,(m(2)*i-lambda(2)):m(2)*i);
    V2(i) = u2_ti*ExpAlmon2;
end
V3=[];
for i = 2:X_len
    u3_ti = hf_data3(1,(m(3)*i-lambda(3)):m(3)*i);
    V3(i) = u3_ti*ExpAlmon3;
end
V = [V1;V2;V3];

% 参数估计
X = LowF_data(2:end)';
time = [2:x_len]'-T(1);
Vt = V-V(:,1);
one = ones(x_len-1,1);
bm_ZLF = ZLF(1:(train_len-1))';
bm_Vt = Vt(:,2:train_len)';
Xi = [time,bm_ZLF,bm_Vt,one];
% par = (Xi'*Xi)\Xi'*X;
% par = pinv(Xi'*Xi)*(Xi'*X);
par = lsqminnorm(Xi'*Xi,Xi'*X);


%% %%%%%%%%%% 模拟生成 %%%%%%%%%%
% 时间响应式
a0 = par(1)/(1-0.5*par(2));
a1 = par(2)/(1-0.5*par(2));
pb = par(3:end-1);                                     % theta
bmB = pb/(1-0.5*par(2));
eta = par(end);
LF_simdata(1) = LowF_data(1); 
for k = 2:X_len
    LF_simdata(k) = a0*(k-1)+a1*LF_cum(k-1)+bmB'*Vt(:,k)+eta;
end


%%  %%%%%%%%%% 模拟和预测精度 %%%%%%%%%%
ape = ((LF_simdata(2:train_len)-lf_data(2:train_len))./lf_data(2:train_len))*100;
APE = ((LF_simdata(1:end)-lf_data(1:end))./lf_data(1:end))*100;
MAPE_train = mean(abs(ape));
MAPE_test = mean(abs(APE(train_len+1:end)));
% LASSO算法
% ENet = 0.5*mean((LF_simdata(2:train_len)-lf_data(2:train_len)).^2)+0.1*norm(par,1);     % 修改正则化参数
ENet = 0.5*mean((LF_simdata(2:end)-lf_data(2:end)).^2)+0.1*norm(par,1);     % 修改正则化参数

% % ENet算法(https://doi.org/10.1016/j.ijforecast.2022.05.005)
% rho = 0.1;                                                        % 修改正则化参数
% ENet = (2*(train_len-1))\sum((LF_simdata(2:train_len)-lf_data(2:train_len)).^2)+...
%     (2*(train_len-1))\rho*(0.25*norm(par,2)^2+0.5*norm(par,1));   


end
