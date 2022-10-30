function [y_sorted,score_sorted] = sort_score(score_matrix,interact_matrix)
%SORT_SCORE 此处显示有关此函数的摘要
%   此处显示详细说明
    [a,b]=size(interact_matrix);
    [B,index] = sort(score_matrix, 'descend');
    index;
    score_sorted = zeros([a,b]);
    y_sorted = zeros([a,b]);
    for i=1:b
        c = score_matrix(:,i);
        d = interact_matrix(:,i);
        score_sorted(:,i) = c(index(:,i));
        y_sorted(:,i) = d(index(:,i));
end

