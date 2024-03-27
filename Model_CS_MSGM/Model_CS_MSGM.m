%  ��������: CS-MSGM(1,N)ģ�Ͳ������Ƽ������Ż�
%          ���ڲ������㷨�Ļ�Ƶ���ݻ�ɫģ�ͣ�����Ӱ�����صĸ�Ƶ��������Ԥ��ϵͳ��Ϊ���ص�Ƶ��������
%  ��������: ��Ƶ����
%  ������: ģ�Ͳ���; ��Ƶ�ͺ�Ȩ�ز���; ��Ƶ����Ԥ��ֵ
%  ע��: �����ļ� fx_MSGM1N.m��initialization.m �뱾��������ͬһĿ¼��

clc,clear all;
format short g;
fprintf(2,'�������㷨�����Ż��Ļ�Ƶ���ݻ�ɫ�����Ԥ��ģ��CS-MSGM(1,N)��\n')

%% R2020b_MATLAB ��������
%% %%%%%%%%%%%%%%%%%%%%%%%% ��.xls�����ļ���ӵ�·�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mkdir('/Users/anyimeng/����/MATLABCode/MATLAB_Mix.data/Table') ;  
% addpath('/Users/anyimeng/����/MATLABCode/MATLAB_Mix.data/Table') ; 
% savepath /Users/anyimeng/����/MATLABCode/MATLAB_Mix.data/Table/pathdef.m ;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % ��Ƶ����ʱ������
sheetNames = sheetnames("China_3E.xlsx");
% �������Ƶ����
lf_data = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","C2:C23"); % ����table
lf_data = lf_data{:,:}';  % table ת matrix  % ������
% �Ա�����Ƶ����1����ȣ�
hf_data3 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","J2:J243"); % ����table
hf_data3 = hf_data3{:,:}'; % table ת matrix  % ������
% �Ա�����Ƶ����2�����ȣ�
hf_data2 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","G2:G89"); % ����table
hf_data2 = hf_data2{:,:}'; % table ת matrix  % ������
% �Ա�����Ƶ����3���¶ȣ�
hf_data1 = readtable('China_3E.xlsx',"ReadRowNames",false,...
    "ReadVariableNames",false,"Sheet",'China',"Range","D2:D23"); % ����table
hf_data1 = hf_data1{:,:}'; % table ת matrix  % ������


%%%%% ʱ�����г��� %%%%% 
X_len = length(lf_data);                          
U1_len = length(hf_data1);
U2_len = length(hf_data2);  
U3_len = length(hf_data3);
U_len = [U1_len,U2_len,U3_len];
m = U_len./X_len;     % ��Ƶ�������������Ƶ�ʱ���

%%%%% ģ�Ͳ������� %%%%% 
object = 3;                                       % ����ָ�����
test_len = 5;                                     % ���Լ�����
train_len = X_len-test_len;                      % ѵ��������
lambda = [1,7,21];

% For reproducibility, set the random seed.
rng(0); % rng default

%% %%%%%%%%%%%%%%% ���ò������㷨ѡ����ѵĶ���ʽ����lambda��theta %%%%%%%%%%%%%%%%%%%
% Change this if you want to get better results
    % Number of nests (or different solutions)
    n = 30;%������
    % Discovery rate of alien eggs/solutions
    pa = 0.25;% Probability of building a new nest(After host bird find exotic bird eggs)
    N_IterTotal = 300;%��������
    %% Simple bounds of the search domain
    nd = 6; %�����ά��,һ�����񵰵ĸ��� Dimensionality of solution
    % Lower bounds
    Lb = -5 * ones(1,nd); %�����½�
    % Upper bounds
    Ub = 5 * ones(1,nd);%�����Ͻ�

    % Random initial solutions
    for i=1:n
        nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb)); %��ʼ����������
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
        %ͨ��levy���в���һ����Generate new solutions by Levy flights
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
            fmin=fnew;%%%%%%%������Ӧ��ֵ
            bestnest=best;%%%%%%%%�������
        end
    end %% End of iterations
%     Monte_Carlo_CS(kk)=fmin;
%% Post-optimization processing
%% Display all the nests
disp(strcat('Total number of iterations=',num2str(N_iter)));
fmin   %%%%������Ӧ��ֵ����ϲ��ֵ�MAPE
bestnest %%%%%%���Ǵ���Ĳ���

%% %%%%%% ���Ǹ�������ֵ����õ�����������ͺ� %%%%%% 
[par,LF_simdata,APE,MAPE_train,MAPE_test,ENet] = fx_MSGM14(lf_data,hf_data1,hf_data2,hf_data3,...
    train_len,lambda,bestnest);     % ����GWOѰ�Ž�� bm_theta = bestnest
Fiting_value = LF_simdata';
Result = [lf_data',LF_simdata',abs(APE)'];

%% ��ʵֵ��Ԥ��ֵ���Ƚ�
figure
plot(lf_data,'bo-','linewidth',1.2)
hold on
plot(LF_simdata,'r*-','linewidth',1.2)
legend('��ʵֵ','ģ��ֵ')
xlabel('�����������'),ylabel('ָ��ֵ')
title('MSGM_{CS}��ʵֵ��ģ��ֵ�ĶԱ�')
set(gca,'fontsize',12)

%�������
error = LF_simdata-lf_data;      %Ԥ��ֵ����ʵֵ�����
APE = (error./lf_data)*100;
MAPE_train = mean(abs(APE(1:train_len)));
MAPE_test = mean(abs(APE(train_len+1:end)));
MAPE = [MAPE_train,MAPE_test];
RMSE_test = sqrt(mean(error(train_len+1:end)*error(train_len+1:end)'));
R2 = 1-sum(error*error')/sum((lf_data-mean(lf_data)).^2);
Theil_U = sqrt(sum(error*error'))/sqrt(sum(lf_data.^2));
r = corrcoef(LF_simdata,lf_data);    %corrcoef�������ϵ�����󣬰�������غͻ����ϵ��

%%%%%% ������
Result = [lf_data', LF_simdata', abs(APE)'];
disp(['num         Actual_value     Fiting_value          APE']);
for i=1:X_len
    disp([i,lf_data(i),LF_simdata(i),abs(APE(i))])
end
MAPE = [MAPE_train,MAPE_test,RMSE_test,R2,Theil_U];
disp('MAPE_train       MAPE_test       RMSE_test       R2        Theil_U');
disp(MAPE);


% %% %%%%%%%%% �ͺ����Ȩ��
% bm_theta = bestnest;
% logwei_expalmon1 = [];          % Expalmon����ʽȨ������
% for i = 1:lambda(1)+1
%     logwei_expalmon1(i,1) = exp(bm_theta(1)*(i-1)+bm_theta(2)*(i-1)^2);
% end
% ExpAlmon1 = logwei_expalmon1/sum(logwei_expalmon1,'all');     % �к�Ϊ1
% 
% logwei_expalmon2 = [];          % Expalmon����ʽȨ������
% for i = 1:lambda(2)+1
%     logwei_expalmon2(i,1) = exp(bm_theta(3)*(i-1)+bm_theta(4)*(i-1)^2);
% end
% ExpAlmon2 = logwei_expalmon2/sum(logwei_expalmon2,'all');     % �к�Ϊ1
% 
% logwei_expalmon3 = [];          % Expalmon����ʽȨ������
% for i = 1:lambda(3)+1
%     logwei_expalmon3(i,1) = exp(bm_theta(5)*(i-1)+bm_theta(6)*(i-1)^2);
% end
% ExpAlmon3 = logwei_expalmon3/sum(logwei_expalmon3,'all');     % �к�Ϊ1


