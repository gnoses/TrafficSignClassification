% load trainComplete

%% compute accuracy on the test and train set
% [~, ~, pred] = costFunc( opt_params, ei, data_test, [], true);
% [~,pred] = max(pred);
% toc();
% acc_test = mean(pred'==labels_test);
% fprintf('test accuracy: %f\n', acc_test);
% f1Test = fmeasure(labels_test', pred);
% fprintf('%s %s test f1: %f\n',actFunc, energyFunc, f1);

[~, ~, pred] = costFunc( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
% acc_train = mean(pred'==labels_train);
% fprintf('train accuracy: %f\n', acc_train);
f1Train = fmeasure(labels_train', pred);
% fprintf('%s %s lambda : %f -> train : %f, test : %f\n',actFunc, energyFunc, lambda, f1Train, f1Test);
fprintf('%s %s lambda : %f -> train : %f\n',actFunc, energyFunc, lambda, f1Train);