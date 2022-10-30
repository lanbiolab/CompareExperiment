clear;
seed = 12345678;
rand('seed', seed);
nfolds = 10; nruns=1;
dataname = '';
% load adjacency matrix
y = importdata('../data/associations2.xls');
% y = h5read('../data/disease-circRNA.h5','/g4/lat');
fold_aupr=[];fold_auc=[];fold_accuracy=[];fold_precision=[];fold_recall=[];fold_F1=[];
fold_tpr=[];fold_fpr=[];

%%IsHG是不是超图，1代表是超�?

% globa_true_y_lp=[];
% globa_predict_y_lp=[];
for run=1:nruns
    % split folds
	[num_D,num_G] = size(y);
    for i=1:num_G
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
        y_train = y_train+y;
        [row,col] = find(y_train==2)
        predict_matrix(row, col) = -20;
        predict_scores;
        [sorted_circrna_disease_matrix, sorted_score_matrix] = sort_score(predict_scores, y_train);
        [c,d]=size(sorted_circrna_disease_matrix);
        tpr_list = [];
        fpr_list = [];
        recall_list = [];
        precision_list = [];
        accuracy_list = [];
        F1_list = [];
        for cutoff=1:c
            P_matrix = sorted_circrna_disease_matrix(1:cutoff, :);
            N_matrix = sorted_circrna_disease_matrix(cutoff+1:c, :);
            TP = sum(P_matrix(:) == 1);
            FP = sum(P_matrix(:) == 0);
            TN = sum(N_matrix(:) == 0);
            FN = sum(N_matrix(:) == 1);
            tpr = TP / (TP + FN);
            fpr = FP / (FP + TN);
            recall_ = TP / (TP + FN);
            precision_ = TP / (TP + FP);
            accuracy_ = (TN + TP) / (TN + TP + FN + FP);
            f1_ = (2 * TP) / (2 * TP + FP + FN);
            tpr_list = [tpr_list,tpr];
            fpr_list = [fpr_list,fpr];
            recall_list = [recall_list,recall_];
            precision_list = [precision_list,precision_];
            accuracy_list = [accuracy_list,accuracy_];
            F1_list = [F1_list,f1_];
        end
%             accuracy_list.append(accuracy);
        if isempty(fold_precision)
            fold_precision=precision_list;
        else
            fold_precision=[fold_precision;precision_list];
        end
        if isempty(fold_recall)
            fold_recall=recall_list;
        else
            fold_recall=[fold_recall;recall_list];
        end
        if isempty(fold_fpr)
            fold_fpr=fpr_list;
        else
            fold_fpr=[fold_fpr;fpr_list];
        end
        if isempty(fold_tpr)
            fold_tpr=tpr_list;
        else
            fold_tpr=[fold_tpr;tpr_list];
        end
        if isempty(fold_accuracy)
            fold_accuracy=accuracy_list;
        else
            fold_accuracy=[fold_accuracy;accuracy_list];
        end
        if isempty(fold_F1)
            fold_F1=F1_list;
        else
            fold_F1=[fold_F1;F1_list];
        end

		
% 		globa_true_y_lp=[globa_true_y_lp;test_labels];
% 		globa_predict_y_lp=[globa_predict_y_lp;predict_scores];
		
		
    
    
    end
end
% RMSE = sqrt(sum((globa_predict_y_lp-globa_true_y_lp).^2)/length(globa_predict_y_lp))

mean_precision = mean(fold_precision,1);
mean_recall = mean(fold_recall,1);
mean_tpr = mean(fold_tpr,1);
mean_fpr = mean(fold_fpr,1);
roc_auc = trapz(mean_fpr, mean_tpr)
aupr_auc = trapz(mean_recall,mean_precision)
mean_accuracy = mean(mean(fold_accuracy, 2), 1)
mean_recall = mean(mean(fold_recall, 2), 1)
mean_precision = mean(mean(fold_precision, 2), 1)
mean_F1 = mean(mean(fold_F1, 2), 1)

% dlmwrite('data5_denovo_fpr.txt',mean_fpr,'delimiter', '\t');
% dlmwrite('data5_denovo_tpr.txt',mean_tpr,'delimiter', '\t');
% dlmwrite('data5_denovo_recall.txt',mean_recall,'delimiter', '\t');
% dlmwrite('data5_denovo_precision.txt',mean_precision,'delimiter', '\t');
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

