function [ rank_result,known_assoccition_rank ] = rankCircRNAs( result,md_adjmat,circRNAs,diseases )

    [rows,cols]=size(md_adjmat);
    num_ones=zeros(cols,1);
    for i=1:cols
       num_ones(i,1)=nnz(md_adjmat(:,i)); %nnz ¨ú¤£?0ªº¤¸¯ÀŸÄ?
    end
    num=rows-min(num_ones);  %min?§ä³Ì¤p­È
    rank_result=cell(num+1,cols);  
    known_assoccition_rank=zeros(size(md_adjmat)); 
    for i=1:cols
       idx=find(md_adjmat(:,i));  %X=find(A)ªð¦^¯x?A¤¤«D¹s¤¸¯À¦ì¸m
       [~,idx_sort]=sort(result(:,i),'descend');
       
       known_assoccition_rank(:,i)=ismember(idx_sort,idx);  
       
       for j=1:length(idx)  
          del_idx= (idx(j,1)==idx_sort);
          idx_sort(del_idx,:)=[];
       end
       rank_result(1,i)=diseases(i,1);  
       for k=1:length(idx_sort)
          rank_result(k+1,i)=circRNAs(idx_sort(k,1)); 
       end
        
    end

end