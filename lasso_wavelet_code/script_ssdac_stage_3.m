clear 
clc
close all

dataFolder = fullfile(cd, '..','Data');
resultsFolder = fullfile(cd, '..','Results');

model_names={
    % 'STD_PLS' % only used for internal comparison               
    'STD_LASSO'                   
    'WAV_LASSO_Haar'              
    'WAV_LASSO_Daubechies_4'      
    'WAV_LASSO_Daubechies_6'      
    'WAV_LASSO_Symmlet_4'         
    'WAV_LASSO_Symmlet_6'         
    % 'FORWARD_PLS'  % only used for internal comparison               
    'FORWARD_WAV_PLS_Haar'        
    'FORWARD_WAV_PLS_Daubechies_4'
    'FORWARD_WAV_PLS_Daubechies_6'
    'FORWARD_WAV_PLS_Symmlet_4'   
    'FORWARD_WAV_PLS_Symmlet_6'   
    %'BACKWARDS_PLS' % only used for internal comparison                    
    'BACKWARDS_WAV_PLS_Haar'        
    'BACKWARDS_WAV_PLS_Daubechies_4'
    'BACKWARDS_WAV_PLS_Daubechies_6'
    'BACKWARDS_WAV_PLS_Symmlet_4'   
    'BACKWARDS_WAV_PLS_Symmlet_6'   
    };

M=length(model_names);
R=20;

KPI=nan(200,R,M);
for i=1:length(model_names),
    filename=['kpi_',model_names{i},'.csv'];
    KPI(:,:,i)=readmatrix(fullfile(resultsFolder,'lasso_wavelets','model_kpi',filename));
end

%% Model comparison =======================================================

pVal=nan(M,M,R);
zVal=nan(M,M,R);
for i=1:M,
   for j=i+1:M,
       for r=1:R,
           [p,~,stats]= signrank(KPI(:,r,i),KPI(:,r,j));
           pVal(i,j,r)=p;
           zVal(i,j,r)=stats.zval;

           pVal(j,i,r)=p;
           zVal(j,i,r)=-stats.zval;
       end
   end
end

alpha=0.05;

nVic=nan(M,R);
nTie=nan(M,R);
nDef=nan(M,R);
for r=1:R,
    for i=1:M,
        % victories
        crit=pVal(i,:,r)<alpha & zVal(i,:,r)<0;
        nVic(i,r)=sum(crit);

        % ties
        crit=pVal(i,:,r)>alpha;
        nTie(i,r)=sum(crit);

        % defeats
        crit=pVal(i,:,r)<alpha & zVal(i,:,r)>0;
        nDef(i,r)=sum(crit);

    end
end

figure
[~,index]=sort(sum(nVic,2)+sum(nTie,2),'descend');
bar([sum(nVic(index,:),2) sum(nTie(index,:),2)], 'stacked')
ylabel('score')
set(gca, 'XTick', 1:M,'XTickLabel',replace(model_names(index),'_','-'))
set(gca,'FontSize',16)
legend('Victories','Ties')


