function [diagTemp] = L21_norm(Mat)
    [rows,cols]=size(Mat);
    diagList=zeros(1,rows);
    for i =1:rows
        add=0;
        for j =1:cols
            add=add+Mat(i,j)^2;
        end
        diagList(1,i)=1/(2*sqrt(add));
    end
    diagTemp=diag(diagList);
end