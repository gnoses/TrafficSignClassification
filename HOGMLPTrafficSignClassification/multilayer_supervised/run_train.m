
function [f1Train, f1Test] = run_train( actFunc , energyFunc, hiddenLayerDim, lambda)

silentMode = 1;
if ~exist('actFunc','var')
  actFunc = 'tanh';
  energyFunc = 'crossEntropy';
  lambda = 0.002;
  silentMode = 0;
  hiddenLayerDim = [256];
end;


%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
% [data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();
% [data_test, labels_test, data_train, labels_train] = load_preprocess_mnist();
% m = 100;
% data_train = data_train(:,1:m);
% labels_train = labels_train(1:m);

load trainData
data_train = double(reshape(data_train,50*50,[])) ./ 255;

ei.input_dim = 50*50;
% number of output classes
ei.output_dim = max(labels_train);
% sizes of all hidden layers and the output layer
ei.layer_sizes = [hiddenLayerDim,  ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = lambda;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
% ei.activation_fun = 'logistic';
ei.activation_fun = actFunc;

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);
% load('opt_params');
% params = opt_params;

% grad_check(@supervised_dnn_cost, params, 10, ei, data_train, labels_train);
% grad_check(@supervised_dnn_cost_sme, params, 10, ei, data_train, labels_train);

%% setup minfunc options
options = [];
options.display = 'iter';
if  (silentMode == 1)
    options.display = 'off';
end
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
if strcmp(energyFunc, 'crossEntropy')
    costFunc = @supervised_dnn_cost;
else
    costFunc = @supervised_dnn_cost_sme;
end


if (1)
    tic();
    [opt_params,opt_value,exitflag,output] = minFunc(costFunc,...
            params,options,ei, data_train, labels_train);
    save('opt_params.mat','opt_params');
    toc();
else
    load('opt_params.mat','opt_params');
end

load testData
data_test = double(reshape(data_test,50*50,[])) ./ 255;

%% compute accuracy on the test and train set
tic();
[~, ~, pred] = costFunc( opt_params, ei, data_test(:,1), [], true);
toc();
[~, ~, pred] = costFunc( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);

acc_test = mean(pred'==labels_test);
%  fprintf('test accuracy: %f\n', acc_test);
f1Test = fmeasure(labels_test', pred);
% fprintf('%s %s test f1: %f\n',actFunc, energyFunc, f1);

[~, ~, pred] = costFunc( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
% fprintf('train accuracy: %f\n', acc_train);
f1Train = fmeasure(labels_train', pred);
fprintf('%s %s , lambda : %f -> train : %.2f, %.2f, test : %.2f %.2f\n',actFunc, energyFunc, lambda, acc_train, f1Train, acc_test,f1Test);