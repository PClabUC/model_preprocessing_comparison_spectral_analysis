function [ kpls, RMSE_k  ] = kSelectPLS_MCCV(X,Y,index_mccv,scaling_method )
% Cross-validation

if nargin<=2,
    index_mccv=[];
end
if nargin<=3,
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


thresh_kpls=0.15;
% thresh_kpls=0.05;

% scaling_method='auto-scaling';
% scaling_method='mean-centering';

nRCV=size(index_mccv.train,2);
[n,m]=size(X);


RMSE_k=zeros(nRCV,m);
PARAM=cell(nRCV,1);
for k=1:m,
    
    for r=1:nRCV,

        indexTrain=index_mccv.train(:,r);
        indexTest=index_mccv.test(:,r);

        %..............................................................

        if k==1,

            switch scaling_method
                case 'auto-scaling'
                    meanX=mean(X(indexTrain,:));
                    stdX=std(X(indexTrain,:));
                    zX=zScale(X,meanX,stdX);

                    meanY=mean(Y(indexTrain,:));
                    stdY=std(Y(indexTrain,:));
                    zY=zScale(Y,meanY,stdY);
                case 'mean-centering'
                    meanX=mean(X(indexTrain,:));
                    stdX=ones(1,m);
                    zX=zScale(X,meanX,stdX);

                    meanY=mean(Y(indexTrain,:));
                    stdY=std(Y(indexTrain,:));
                    zY=zScale(Y,meanY,stdY);
            end


            [ P, Q, B, W, BETA, RES ] = plsModel( zX(indexTrain,:), zY(indexTrain,:), 1 );
            par.zXtest=zX(indexTest,:);
            par.meanY=meanY;
            par.stdY=stdY;
            par.P=P;
            par.Q=Q;
            par.B=B;
            par.W=W;
            par.X=RES.X;
            par.Y=RES.Y;
            PARAM{r}=par;

        else
            par=PARAM{r};

            [ P, Q, B, W, BETA, RES ] = plsModel( par.X, par.Y, 1 );
            par.P=[par.P P];
            par.Q=[par.Q Q];
            par.B=[par.B B];
            par.W=[par.W W];
            par.X=RES.X;
            par.Y=RES.Y;
            PARAM{r}=par;
        end

        B=diag(par.B);
        BETA.T=par.W*inv(par.P'*par.W);
        BETA.Y=BETA.T*B*par.Q';

        %..............................................................

        Yhat=par.zXtest*BETA.Y*par.stdY+par.meanY;

        E=Y(indexTest,:)-Yhat;
        RMSE_k(r,k)=sqrt(mean(E.^2));

    end
   
    
    if k>1 && k~=m,

        % [P,H] = signrank(...,'tail',TAIL) performs the test against the
        %     alternative hypothesis specified by TAIL:
        %      'both'  -- "median is not zero (or M)" (two-tailed test, default)
        %      'right' -- "median is greater than zero (or M)" (right-tailed test)
        %      'left'  -- "median is less than zero (or M)" (left-tailed test)

        % right: H0: RMSE_ref <= RMSE_tent(:,k)
        % right: H1: RMSE_ref > RMSE_tent(:,k)
        pVal= signrank(RMSE_k(:,k-1),RMSE_k(:,k),'tail','right');% For a two-sample test, the alternate hypothesis states the data in x – y come from a distribution with median less than 0.
        if pVal>thresh_kpls,
            kpls=k-1;
            break
        end

    else
        kpls=m;
    end
    
end

RMSE_k=RMSE_k(:,1:kpls);


end

