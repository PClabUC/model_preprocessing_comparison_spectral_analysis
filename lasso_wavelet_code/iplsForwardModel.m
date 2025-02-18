function [model] = iplsForwardModel(X, Y, index_mccv, nInter, scaling_method)
% 14.04.2023

if nargin<=2,
    index_mccv=[];
end
if nargin<=3,
    nInter=10;
end
if nargin<=4,
    scaling_method='auto-scaling';
end

if isempty(index_mccv)==1 || isstruct(index_mccv)==0,,

    nRCV=200;%20;% number of cross-validation replicates

    n=size(X,1);
    nC=floor(0.80*n);

    % Set random permutations
    index_mccv.train=zeros(nC,nRCV);
    index_mccv.test=zeros(n-nC,nRCV);
    for ii=1:nRCV,
        ind_rnd=randperm(n,n);
        index_mccv.train(:,ii)=ind_rnd(1:nC);
        index_mccv.test(:,ii)=ind_rnd(nC+1:end);
    end
else
    nRCV=size(index_mccv.train,2);
end

threshold=0.10;

kmax=size(X,2);


if length(nInter)==1
    breakPoints=round(linspace(1,kmax+1,nInter+1));

    varIndex=1:kmax;
    intervalIndex=nan(1,kmax);
    for i=1:nInter
        intervalIndex(varIndex>=breakPoints(i) & varIndex<breakPoints(i+1))=i;
    end

else
    intervalIndex=nInter;
    nInter=max(intervalIndex);
end

% Initial model -----------------------------------------------------------

RMSE_tent=nan(nRCV,nInter);

for i=1:nInter,
    
    model_tent=[];

   inModel_tent=intervalIndex==i;
 
    [ kpls, RMSE_aux ] = kSelectPLS_MCCV( X(:,inModel_tent), Y, index_mccv, scaling_method  );       
    [ model_tent.P, model_tent.Q, model_tent.B, model_tent.W, model_tent.BETA ] = plsModel( X(:,inModel_tent), Y, kpls );
    model_tent.k=kpls;
    model_tent.inModel=inModel_tent;
    RMSE_tent(:,i)=RMSE_aux(:,end);
                
    if i==1,
        model_ref=model_tent;
        RMSE_ref=RMSE_tent(:,i);
        med_RMSE=median(RMSE_ref);
        int_ref=1;
    else
        if median(RMSE_tent(:,i))<med_RMSE
            model_ref=model_tent;
            RMSE_ref=RMSE_tent(:,i);
            med_RMSE=median(RMSE_ref);
            int_ref=i;
        end
    end
    
end

%--------------------------------------------------------------------------

% Forward addition --------------------------------------------------------

inModel=intervalIndex==int_ref;
inModel_var=int_ref;
for i=1:nInter,
    
    pVal=nan(nInter,1);
    RMSE_tent=nan(nRCV,nInter);
    med_RMSE=nan(nInter,1);
    model_tent=cell(nInter,1);
    k_range=setdiff(1:nInter,inModel_var);
    parfor k=1:nInter,
        if sum(k_range==k)==1,

            model_aux=[];
            inModel_tent=inModel | intervalIndex==k;

            [ kpls, RMSE_aux] = kSelectPLS_MCCV( X(:,inModel_tent), Y, index_mccv, scaling_method );
            RMSE_tent(:,k)=RMSE_aux(:,end);
            [ model_aux.P, model_aux.Q, model_aux.B, model_aux.W, model_aux.BETA ] = plsModel( X(:,inModel_tent), Y, kpls );
            model_aux.k=kpls;
            model_aux.inModel=inModel_tent;

            model_tent{k}=model_aux;

            % [P,H] = signrank(...,'tail',TAIL) performs the test against the
            %     alternative hypothesis specified by TAIL:
            %      'both'  -- "median is not zero (or M)" (two-tailed test, default)
            %      'right' -- "median is greater than zero (or M)" (right-tailed test)
            %      'left'  -- "median is less than zero (or M)" (left-tailed test)

            % right: H0: RMSE_ref <= RMSE_tent(:,k)
            % right: H1: RMSE_ref > RMSE_tent(:,k)
            pVal(k)= signrank(RMSE_ref,RMSE_tent(:,k),'tail','right');% For a two-sample test, the alternate hypothesis states the data in x - y come from a distribution with median less than 0.
            med_RMSE(k)=median(RMSE_tent(:,k)); 

        end
    end
    
    % interval can enter in the model if RMSE_ref > RMSE_tent(:,k):
    % that is, if H0 is rejected:
    can_enter=pVal<threshold;
    if sum(can_enter)>=1,
        % add the best
        med_RMSE(~can_enter)=inf;
        [~,indexSelect]=min(med_RMSE);
        
        model_ref=model_tent{indexSelect};
        RMSE_ref=RMSE_tent(:,indexSelect);
        
        inModel=inModel | intervalIndex==indexSelect;
        inModel_var=[inModel_var indexSelect];
        
    else
        break
    end
    
end

model=model_ref;
model.inModel_var=inModel_var;

%--------------------------------------------------------------------------

end

