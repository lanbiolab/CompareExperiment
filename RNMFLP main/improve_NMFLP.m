function [P] = improve_NMFLP(A,circSimi,disSimi,beita,gama,k,iterate)
    %ALP=A;
    [rows,cols]=size(A);
    % 初始化circRNA和disease迭代矩阵
    C = abs(rand(rows,k));
    D = abs(rand(cols,k));
    
    % 计算对角矩阵
    diag_cf=diag(sum(circSimi,2));
    diag_df=diag(sum(disSimi,2));
     % 避免假阴性数据对矩阵分解的影响，因此对邻接矩阵进行重构
    A1=circSimi*A;
    A2=A*disSimi;
    for i=1:rows
        for j=1:cols
            A(i,j)=max(A1(i,j),A2(i,j));
        end
    end
    
    % 归一化
    AA=zeros(rows,cols);
    for j =1:cols
        colList=A(:,j);
        for i =1:rows
        AA(i,j)=(A(i,j)-min(colList))/(max(colList)-min(colList));
        end
    end
    A=AA;
    
    % 鲁棒性矩阵分解部分
    for step = 1:iterate
        Y=A-C*D';
        B=L21_norm(Y);
        
        % for circRNA coding
        if  beita >0
            BAD=B*A*D+beita*circSimi*C;
            BCDD=B*C*(D')*D+beita*diag_cf*C;
        end
        C=C.*(BAD./BCDD);
        
        %for disease coding
        if  beita >0
            ABC=(A')*B*C+beita*disSimi*D;
            DCBC=D*(C')*B*C+beita*diag_df*D;
        end
        D=D.*(ABC./DCBC);
        
        scoreMat_NMF=A-C*D';
        error=mean(mean(abs(scoreMat_NMF)))/mean(mean(A));
        fprintf('step=%d  error=%f\n',step,error);
    end
     save C C;
     save D D;
    % 标签传播
    load C C;
    load D D;
    Ac=A;
    Mc=C*D';
    Md=Mc';
    Ad=A';
    
    FC=A;
    FD=A';
    FC0=(Mc+Ac)/2;
    FD0=(Md+Ad)/2;
    K=0;
    
    % 相似性矩阵归一化
    CCSimi=zeros(rows,rows);
    CC=sum(circSimi);
    for i =1:rows
        for j =1:rows
            CCSimi(i,j) = circSimi(i,j)/(((CC(i)*CC(j))^0.5));
        end
    end
    DDSimi=zeros(cols,cols);
    DD=sum(disSimi);
    for i =1:cols
        for j =1:cols
            DDSimi(i,j) = disSimi(i,j)/(((DD(i)*DD(j))^0.5));
        end
    end
    
    
    delta=1;
    while (delta > 1e-6)
        FC1 = gama*CCSimi*FC+ (1-gama)*FC0;
        delta =abs(sum(sum((abs(FC1)-abs(FC)))));
        FC=FC1;
        K=K+1;
    end
    
    delta=1;
    while (delta > 1e-6)
        FD1 = gama*DDSimi*FD+ (1-gama)*FD0;
        delta =abs(sum(sum((abs(FD1)-abs(FD)))));
        FD=FD1;
        K=K+1;
    end
    
    P=0.5*(FC1+FD1');
end