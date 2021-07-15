function result = func_CRDBPSW(Data, win_out, win_in,lambda)
%% CRD based background purification[with Least Square method] and saliency weight( CRDBPSW)
% Compiled by Zephyr Hou on 2019-12-25
%% Instruction:
%  Step1: Using the Least Square Method to get the intial coefficient.
%  Step2: Sorting coefficient from large to small.
%  Step3: Obtain the number of last reserved pixels.
%  Step4: Calculate saliency weight map.
%  Step5: Collaborative Representation Detection.
%  Step6: Obtain the final probability map.
%% Function Usage:
%   [result] = func_CRD_LR_SW(Data, window)
% Inputs
%   Data - 3D data matrix (num_row x num_col x num_dim)
%   window - spatial size window (e.g., 3, 5, 7, 9,...)
% Outputs
%   result - Detector output (num_row x num_col)
%%  Main Function
[rows,cols,bands] = size(Data);        
result = zeros(rows, cols);
t = fix(win_out/2);
t1 = fix(win_in/2);
M = win_out^2;
num_sam=win_out*win_out-win_in*win_in;
%% Expanding the edges(two methods)
% padding avoid edges
DataTest = zeros(3*rows, 3*cols, bands);
DataTest(rows+1:2*rows, cols+1:2*cols, :) = Data;
DataTest(rows+1:2*rows, 1:cols, :) = Data(:, cols:-1:1, :);
DataTest(rows+1:2*rows, 2*cols+1:3*cols, :) = Data(:, cols:-1:1, :);
DataTest(1:rows, :, :) = DataTest(2*rows:-1:(rows+1), :, :);
DataTest(2*rows+1:3*rows, :, :) = DataTest(2*rows:-1:(rows+1), :, :);

% padding zeros to avoid edges
% DataTest = zeros(3*a, 3*b, c);
% DataTest(a+1:2*a, b+1:2*b, :) = Data;
% DataTest(a+1:2*a, 1:b, :) = zeros(a,b,c);
% DataTest(a+1:2*a, 2*b+1:3*b, :) = zeros(a,b,c);
% DataTest(1:a,:,:)=zeros(a,3*b,c);
% DataTest(2*a+1:3*a,:,:)=zeros(a,3*b,c);
%%
Res_SW=zeros(rows,cols);
for i = 1+rows:2*rows 
    for j = 1+cols:2*cols
        block = DataTest(i-t: i+t, j-t: j+t, :);
        y = squeeze(DataTest(i, j, :)).';% 1 x num_dim
        block(t-t1+1:t+t1+1, t-t1+1:t+t1+1, :) = NaN;
        block = reshape(block, M, bands);
        block(isnan(block(:, 1)), :) = [];
        Xs = block';  % num_dim x num_sam 
        
       %% 显著性权重  
        LocPixs=DataTest(i-t1:i+t1,j-t1:j+t1,:);
        LocPixs=reshape(LocPixs,win_in*win_in,bands)';       % bands x num_in    
        % --------------光谱距离的选择方式---------------- 
        % 光谱角距离
        temp=repmat(y',[1,size(LocPixs,2)]);                 % bands x num_in     
        fz=sum(temp.*LocPixs);                               % 1 x num_in
        fm=sqrt(sum(temp.^2).*sum(LocPixs.^2));              % 1 x num_in
        dis_Ang=acos(fz./fm);                                % 光谱角距离  1 x num_in 
        dis_Spectrum=dis_Ang;                                % 1 x nums_in
              
        %%
        % 欧式距离     
        % dis_Euc=sqrt(sum((temp-Xs).^2));   % 欧式距离  1 x num_in
        % dis_Spectrum=dis_Euc;              % 采用欧式距离效果较好 
        % -----------------------------------------------   
        
        % 位置距离(坐标之间的欧式距离)
        tempWinX=zeros(win_in,win_in);
        tempWinY=zeros(win_in,win_in);
        for ii=1:win_in
            tempWinX(ii,:)=-t1:t1;
            tempWinY(:,ii)=t1:-1:-t1;
        end
        dis_Position=sqrt(tempWinX.^2+tempWinY.^2);        % num_in x mum_in
        dis_Pos=reshape(dis_Position,win_in*win_in,1)';    % 1 x num_in
        
        Const=1;
        dis_Sal=dis_Spectrum./(1+Const*dis_Pos);           % 1 x num_in 
        dis_Sal_ava=sum(dis_Sal)/(size(LocPixs,2)-1);
        
        Res_SW(i-rows,j-cols)=dis_Sal_ava;       
                   
        %% 最小二乘法(添加和为1约束)
        tempX=[ones(bands,1),Xs];

        % 方法一：直接求
%         paraV=pinv(tempX'*tempX)*tempX'*y';  
        % 方法二：直接求逆运算有时候出错，此处使用svd奇异值分解求逆
        paraV=func_pseudoInv(tempX'*tempX)*tempX'*y';
        
        H = block';  % num_dim x num_sam
        temp1=sum(H);
        miu=mean(temp1);
        sigma=std(temp1);
        threshold_max=miu+2*sigma;
        threshold_min=miu-2*sigma;
        for ii=1:size(H,2)
            if temp1(ii) < threshold_min || temp1(ii) > threshold_max
                temp1(ii)=NaN;
            end
        end
        H(:,isnan(temp1(:))) = [];
        num1=size(H,2);
        paraV(1)=[];  % 常数去掉
        [~,ind]=sort(paraV,'descend');
        ind=ind(1:num1);       
        Xs=Xs(:,ind);
       %% 
        num_sam=size(Xs,2);
        Gamma=zeros(1,num_sam);
   
        y_1=[y,1];                  % 1 x (num_dim + 1 )
        Xs_1=[Xs;ones(1,num_sam)];  % (num_dim + 1) x num_sam
        
        for k=1:num_sam
            Gamma(1,k)=norm((y'-Xs(:,k)),2);
        end
        
        Gamma_y=diag(Gamma);  % num_sam x num_sam    
        %% 矩阵求逆方法一：直接求
%         weights=pinv(Xs_1'*Xs_1+lambda*(Gamma_y'*Gamma_y))*Xs_1'*y_1'; % num_sam x 1  (formula 8)
       
        %% 矩阵求逆方法二：SVD方法求伪逆
         tmpMat=Xs_1'*Xs_1+lambda*(Gamma_y'*Gamma_y);
         weights=func_pseudoInv(tmpMat)*Xs_1'*y_1'; % num_sam x 1  (formula 8)               
        %%       
        y_hat = (Xs*weights(:))';  % 1 x num_dim
        result(i-rows, j-cols) = norm(y - y_hat, 2);        
        
    end
end

result=(result-min(result(:)))/(max(result(:))-min(result(:)));
Res_SW=(Res_SW-min(Res_SW(:)))/(max(Res_SW(:))-min(Res_SW(:)));

result=result.*Res_SW;

% figure;
% subplot(1,2,1);imshow(Data(:,:,100),[]);
% subplot(1,2,2);imshow(Res_SW,[])
end