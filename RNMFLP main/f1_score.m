function [accurary, precision, recall, f1] = f1_score(label, predict)
   [M,~] = confusionmat(label, predict);
   [a,b] = size(M)
   %以下两行为二分类时用
%    TPR = M(2,2) / (M(2,1) + M(2,2)); %SE: TP/(TP+FN)
%    TNR = M(1,1) / (M(1,1) + M(1,2)); %SP: TN/(TN+FP)
   TN=M(2,2); FN=M(1,2); FP=M(2,1); TP=M(1,1);
   %转置，可以不转同时调换方向
   M = M';
   precision = diag(M)./(sum(M,2) + 0.0001);  %按列求和: TP/(TP+FP)
   recall = diag(M)./(sum(M,1)+0.0001)'; %按行求和: TP/(TP+FN)
   precision = mean(precision);
   recall = mean(recall);
   f1 = 2*precision*recall/(precision + recall);
%    FPR = FP/(FP+TN);   %定义式
%    TPR = TP/(TP+FN);
   accurary = (TP+TN)/(TP+TN+FP+FN);
%    precision = TP/(TP+FP);
%    recall = TP/(TP+FN);
%    TPR = TP/(TP+FN);
%    FPR = FP/(TN+FP);
%    f1 = 2*precision*recall/(precision+recall);
end
