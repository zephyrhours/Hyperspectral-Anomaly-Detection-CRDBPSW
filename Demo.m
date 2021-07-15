%% This is a demo for CRDBPSW method
% Author: Zengfu Hou
% Reference: 
%     1. Collaborative representation with background purification and
%     saliency weight for hyperspectral anomaly detection.
%     2. A background refinement collaborative representation method with
%     saliency weight for for hyperspectral anomaly detection.
%%
clc;clear;close all;
load('Sandiego.mat');
[rows,cols,bands]=size(hsi);
label_value=reshape(hsi_gt,1,rows*cols);
%%  Normalized
% hsi=(hsi-max(hsi(:)))/(max(hsi(:))-min(hsi(:)));

%% Proposed CRDBPSW Method
disp("Running, please wait....")
tic
R1 = func_CRDBPSW(hsi, 25, 15,1e-6);   %(SanDiego,25,15)(SpecTIR,17,15)
t1=toc;

R1value = reshape(R1,1,rows*cols);
[FA1,PD1] = perfcurve(label_value,R1value,'1') ;
AUC1=-sum((FA1(1:end-1)-FA1(2:end)).*(PD1(2:end)+PD1(1:end-1))/2)

%% =============================================================
clc;
disp('-------------------------------------------------------------------')
disp('CRDBPSW')
disp(['AUC:     ',num2str(AUC1),'          Time:     ',num2str(t1)])
disp('-------------------------------------------------------------------')

figure(1);
semilogx(FA1, PD1, 'k-', 'LineWidth', 2);  hold on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('CRDBPSW','location','southeast')









 