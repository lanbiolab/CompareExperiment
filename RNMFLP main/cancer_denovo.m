clear;
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=1;
dataname = '';
% load adjacency matrix
y = importdata('../data/associations5.xls');
% y = h5read('../data/disease-circRNA.h5','/g4/lat');
fold_aupr=[];fold_auc=[];fold_accurary=[];fold_precision=[];fold_recall=[];fold_F1=[];
fold_tpr=[];fold_fpr=[];

%%IsHG是不是超图，1代表是超图

% globa_true_y_lp=[];
% globa_predict_y_lp=[];
for run=1:nruns
    % split folds
	[num_D,num_G] = size(y);
    for i=29
% 	crossval_idx = crossvalind('Kfold',y(:),nfolds);
%    
% 
%     for fold=1:nfolds
%         train_idx = find(crossval_idx~=fold);
%         test_idx  = find(crossval_idx==fold);

        y_train = y;
        
		y_train(:,i) = 0;
        
		
		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
        %% 4. evaluate predictions
        yy=y;

		
% 		test_labels = yy(:,i);
%         [a,b]=size(test_idx);
		predict_scores = RNMFLP_predict(y_train,80,0.001,0.7,4);
        y_train = y_train*2;
        [row,col] = find(y_train==2);
        predict_matrix(row, col) = -20;
        predict_scores;
        [sorted_circrna_disease_matrix, sorted_score_matrix] = sort_score(predict_scores, y);
        [c,d]=size(sorted_circrna_disease_matrix);
        top10 = length(find(sorted_circrna_disease_matrix(1:10,i)~=0))
        top20 = length(find(sorted_circrna_disease_matrix(1:20,i)~=0))
        top50 = length(find(sorted_circrna_disease_matrix(1:50,i)~=0))
        top100 = length(find(sorted_circrna_disease_matrix(1:100,i)~=0))
        top150 = length(find(sorted_circrna_disease_matrix(1:150,i)~=0))
        top200 = length(find(sorted_circrna_disease_matrix(1:200,i)~=0))
        dlmwrite('cancer.txt',sorted_circrna_disease_matrix,'delimiter','\t');
%         for cutoff=1:c
%             P_matrix = sorted_circrna_disease_matrix(1:cutoff, :);
%             N_matrix = sorted_circrna_disease_matrix(cutoff+1:c, :);
%             TP = sum(P_matrix(:) == 1);
%             FP = sum(P_matrix(:) == 0);
%             TN = sum(N_matrix(:) == 0);
%             FN = sum(N_matrix(:) == 1);
%             tpr = TP / (TP + FN);
%             fpr = FP / (FP + TN);
%             recall_ = TP / (TP + FN);
%             precision_ = TP / (TP + FP);
%             accuracy_ = (TN + TP) / (TN + TP + FN + FP);
%             f1_ = (2 * TP) / (2 * TP + FP + FN);
% %             accuracy_list.append(accuracy);
%             fold_precision=[fold_precision;precision_];
%             fold_F1=[fold_F1;f1_];
%             fold_recall=[fold_recall;recall_];
%             fold_accurary=[fold_accurary;accuracy_];
%             fold_tpr=[fold_tpr;tpr];
%             fold_fpr=[fold_fpr;fpr];
% 
% 		
% % 		globa_true_y_lp=[globa_true_y_lp;test_labels];
% % 		globa_predict_y_lp=[globa_predict_y_lp;predict_scores];
% 		
% 		
%         end
%     
    
    end
end
% RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))


% dlmwrite('data5_denovo_fold_fpr.txt',fold_fpr,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_tpr.txt',fold_tpr,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_recall.txt',fold_recall,'delimiter', '\t');
% dlmwrite('data5_denovo_fold_precision.txt',fold_precision,'delimiter', '\t');
% mean_aupr = mean(fold_aupr)
% mean_auc = mean(fold_auc)
% mean_accurary = mean(fold_accurary)
% mean_precision = mean(fold_precision)
% mean_recall = mean(fold_recall)
% mean_f1 = mean(fold_F1)
% mean_tpr = mean(fold_tpr)
% mean_fpr = mean(fold_fpr)

