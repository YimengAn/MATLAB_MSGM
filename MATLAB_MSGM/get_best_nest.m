%% --------------- All subfunctions are list below -----------------
%% Find the current best nest
function [fmin,best,nest,fitness]=get_best_nest...
    (lf_data,hf_data1,hf_data2,hf_data3,train_len,lambda,nest,newnest,fitness);
% Evaluating all new solutions
X0 = lf_data;
X1 = hf_data1;
X2 = hf_data2;
X3 = hf_data3;
for j=1:size(nest,1)
%     fnew=fobj(newnest(j,:));
    [par,LF_simdata,APE,MAPE_train,MAPE_test,fnew] = fx_MSGM14(...
        lf_data,hf_data1,hf_data2,hf_data3,train_len,lambda,newnest(j,:));

    if fnew<=fitness(j)
       fitness(j)=fnew;
       nest(j,:)=newnest(j,:);
    end
    
end
% Find the current best
[fmin,k]=min(fitness) ;
best=nest(k,:);
% end