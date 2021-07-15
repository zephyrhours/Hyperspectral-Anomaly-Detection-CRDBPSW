function InvCov= func_pseudoInv(imgVec,OptPara)
%% ������α������
% Author: Zephyr Hou
% Time: 2019-11-29
%% Function Usage
% Input:
%     imgVec -- The input matrix
%     OptPara --  Optinal parameters,the default value is 1
% Output:
%     InvCov -- The preudo inverse matrix of imgVec 
%% Main Function

if nargin==1
    OptPara = 1;
elseif OptPara~=fix(OptPara)
    error('MATLAB:func_pseudoInv:WrongNValue',...
        'The value of opt can only be a strictly positive integer 1 or 2!')
end
    

if OptPara == 1
    %% ����һ�����ɷַ���������α�棨ͬ������Ч����ͬ��
    % �˴��൱��PCA�任��ȡǰPCs������ֵ����������
    [eig_XL,eig_Z]=eig(imgVec);
    [Deig_Z,ind]=sort(diag(eig_Z),'descend');
    D_eigXL=eig_XL(:,ind');

    % �Զ�ȷ��ѡ������ɷָ���
    rate = 0.9999;%�ò����ɵ�   
    Sumva1 = rate * sum(Deig_Z); %���ܺ�0.99999������Сȡ������ֵ
    T0=cumsum(Deig_Z);           % cumsumΪ�ۼӺ����������ۼ�  
    ki=find(T0>Sumva1);   
    PCs=ki(1);

    InvCov=D_eigXL(:,1:PCs)*inv(diag(Deig_Z(1:PCs)))*D_eigXL(:,1:PCs)';

  
elseif OptPara == 2
    %% ������������ֵ�ֽ���α��(����֤������)
    % �˴��൱��PCA�任
    [U_tmp,S_tmp,V_tmp]=svd(imgVec);    
    % PCs=size(S_tmp,2);              % S_tmpΪ����ֻ��ʾ����������ɷ���   
    rate = 0.9999;  %�ò����ɵ�
    Sumva1 = rate * sum(diag(S_tmp)); %���ܺ�0.99999������Сȡ������ֵ
    T0=cumsum(diag(S_tmp));           % cumsumΪ�ۼӺ����������ۼ�  
    ki=find(T0>Sumva1);
    
    PCs=ki(1);
    inv_S_tmp=S_tmp(1:PCs,1:PCs);        % num_Pcs x  num_Pcs                
    inv_S_tmp=diag(ones(PCs,1)./diag(inv_S_tmp));   % num_Pcs x  num_Pcs 
    InvCov=V_tmp(:,1:PCs)*(inv_S_tmp)*U_tmp(:,1:PCs)'; % num_Pcs x  num_Pcs
end

end

