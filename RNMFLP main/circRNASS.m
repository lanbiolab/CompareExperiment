
function [ result ] = circRNASS( md_adjmat, disease_ss )
%MIRNASS Summary of this function goes here
%   Detailed explanation goes here
   

rows=size(md_adjmat,1);
result=zeros(rows,rows);

for i=1:rows
    
    idx = find(md_adjmat(i,:)==1);
    
    if isempty(idx)
        continue;
    end
    
    for j=1:i
        
        idy = find(md_adjmat(j,:)==1); %返回第j行的列索引
        
        if isempty(idy)
        continue;
        end
        
        sum1=0;
        sum2=0;
        
        for k1=1:size(idx,2)
            
            sum1=sum1+max(disease_ss(idx(1,k1),idy));
            
        end
        
        for k2=1:size(idy,2)
            
            sum2=sum2+max(disease_ss(idx,idy(1,k2)));
            
        end
        
         result(i,j)=(sum1+sum2)/(k1+k2);
         
         result(j,i)=result(i,j);
            
    end
    
    for k=1:rows
       result(k,k)=1; 
    end
    
end
  

end

