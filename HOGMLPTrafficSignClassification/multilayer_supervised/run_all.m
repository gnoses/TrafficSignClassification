clc; clear all;
lambda = [0.0001 0.001 0.005 0.01]; % 0.05 0.1 0.5 1 5 10];
hiddenLayerDim = [256 512];
for j=1:length(hiddenLayerDim)
        disp (hiddenLayerDim(j));
        for i=1:length(lambda)    
            disp (lambda(i));
    
%         run_train('logistic','crossEntropy', hiddenLayerDim(j), lambda(i));
        run_train('tanh','crossEntropy', hiddenLayerDim(j),  lambda(i));
%         run_train('relu','crossEntropy', hiddenLayerDim(j),  lambda(i));

%         run_train('logistic','sme', hiddenLayerDim(j),  lambda(i));
%         run_train('tanh','sme', hiddenLayerDim(j),  lambda(i));
%         run_train('relu','sme', hiddenLayerDim(j),  lambda(i));
    end
end
