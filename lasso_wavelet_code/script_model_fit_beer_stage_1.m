clear
clc
close all

dataFolder = fullfile(cd, '..','Data');
resultsFolder = fullfile(cd, '..','Results');

%% Load data ==============================================================

Xraw=readmatrix(fullfile(dataFolder, 'beer_spectra.csv'));
Xtrain=Xraw(:,1:end-1);
Ytrain=Xraw(:,end);

Xraw=readmatrix(fullfile(dataFolder, 'beer_spectra_test.csv'));
Xtest=Xraw(:,1:end-1);
Ytest=Xraw(:,end);

lambda=readmatrix(fullfile(dataFolder, 'wavelenghts.csv'));
lambda_label='\nu (cm^{-1})';

clear Xraw

%==========================================================================

% Generate random datasets for cross-validation ---------------------------

% k_folds=4;
nMCCV=400;
perSplit=0.80;
n=size(Xtrain,1);
nC=floor(perSplit*n);
% Set random permutations
rndIndex.train=zeros(nC,nMCCV);
rndIndex.test=zeros(n-nC,nMCCV);
for ii=1:nMCCV,
    ind_rnd=randperm(n,n);
    rndIndex.train(:,ii)=ind_rnd(1:nC);
    rndIndex.test(:,ii)=ind_rnd(nC+1:end);
end

% -------------------------------------------------------------------------

% Generate random datasets for cross-validation - LASSO -------------------

kFold=10;
cvp = cvpartition(length(Ytrain),"KFold",kFold);
lambda_min=1e-4;
lambda_max=0.5;
numLambda=100;
r=(lambda_max/lambda_min)^(1/(numLambda-1));
lasso_lambda=1e-5*r.^(0:numLambda-1);

%--------------------------------------------------------------------------

%==========================================================================

%% Wavelet decomposition ==================================================

Jdec=5;

% train data set ----------------------------------------------------------

wav_case={'Haar';
    'Daubechies_4';
    'Daubechies_6';
    'Symmlet_4';
    'Symmlet_6'};

wname={'haar';
    'db4';
    'db6';
    'sym4';
    'sym6'};

for wav_index=1:5,

    [ W, wavBK, w_coef_number_k, w_coef_number_s ] = waveletDecomp( Xtrain, Jdec, wname{wav_index} );
    [ Xrec ] = waveletReconstruct( W, wavBK, wname{wav_index} );

    Wmat_train.(wav_case{wav_index})=W;
    Wmat_coef_number_k.(wav_case{wav_index})=w_coef_number_k;
    Wmat_coef_number_s.(wav_case{wav_index})=w_coef_number_s;
    Wmat_bk.(wav_case{wav_index})=wavBK;
    Wmat_wname.(wav_case{wav_index})=wname{wav_index};

end

%--------------------------------------------------------------------------

% test dataset ------------------------------------------------------------

for wav_index=1:5,

    [ W, wavBK, w_coef_number_k, w_coef_number_s ] = waveletDecomp( Xtest, Jdec, wname{wav_index} );
    [ Xrec ] = waveletReconstruct( W, wavBK, wname{wav_index} );

    Wmat_test.(wav_case{wav_index})=W;

end

%--------------------------------------------------------------------------

%==========================================================================


%% Train model ============================================================

% standard-PLS ------------------------------------------------------------
model_name='STD_PLS';

model.(model_name).mew_x=mean(Xtrain);
zX=Xtrain-model.(model_name).mew_x;

model.(model_name).mew_y=mean(Ytrain);
model.(model_name).sigma_y=std(Ytrain);
zY=(Ytrain-model.(model_name).mew_y)/model.(model_name).sigma_y;

scaling_method='mean-centering';
[ kpls, RMSE_k ] = kSelectPLS_MCCV( zX, zY, rndIndex,scaling_method );

figure
boxplot(RMSE_k)

kpls=4;% overide automatic selection: too many LV
[ model.(model_name).P, model.(model_name).Q, model.(model_name).B, model.(model_name).W, model.(model_name).BETA ] = plsModel( zX, zY, kpls );
model.(model_name).beta0=model.(model_name).BETA.Y;
model.(model_name).k=kpls;

Yhat=zX*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
[ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

figure
plot(Yhat,Ytrain,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
set(gca,'FontSize',16)

figure
stairs(lambda,model.(model_name).beta0)
xlabel(lambda_label)
ylabel('b')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

% Standard LASSO ----------------------------------------------------------

model_name='STD_LASSO';

model.(model_name).mew_x=mean(Xtrain);
zX=Xtrain-model.(model_name).mew_x;

model.(model_name).mew_y=mean(Ytrain);
model.(model_name).sigma_y=std(Ytrain);
zY=(Ytrain-model.(model_name).mew_y)/model.(model_name).sigma_y;

[B,FitInfo] = lasso(zX, zY,'Standardize',false,'CV',cvp,'Lambda',lasso_lambda);
model.(model_name).FitInfo=FitInfo;

figure
lassoPlot(B,FitInfo,PlotType="CV");
legend("show")

idxLambdaMinMSE = FitInfo.IndexMinMSE;
% minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0);
model.(model_name).LambdaMinMSE=FitInfo.LambdaMinMSE;
model.(model_name).b=B(:,idxLambdaMinMSE);
model.(model_name).b0 = FitInfo.Intercept(idxLambdaMinMSE);
model.(model_name).beta0=model.(model_name).b;

Yhat=(model.(model_name).b0+zX*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;
[ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

figure
plot(Yhat,Ytrain,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

figure
stairs(lambda,model.(model_name).beta0)
xlabel(lambda_label)
ylabel('b')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

% Wavelet-LASSO -----------------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,

        model_name=['WAV_LASSO_',scaling_option{s},'_',(wav_case{wav_index})];

        switch scaling_option{s},
            case 'MC'
                model.(model_name).mew_x=mean(Wmat_train.(wav_case{wav_index}));
                model.(model_name).sigma_x=ones(1,size(Wmat_train.(wav_case{wav_index}),2));
            case 'AS'
                model.(model_name).mew_x=mean(Wmat_train.(wav_case{wav_index}));
                model.(model_name).sigma_x=std(Wmat_train.(wav_case{wav_index}));
        end

        zX=zScale(Wmat_train.(wav_case{wav_index}),model.(model_name).mew_x,model.(model_name).sigma_x);

        model.(model_name).mew_y=mean(Ytrain);
        model.(model_name).sigma_y=std(Ytrain);
        zY=(Ytrain-model.(model_name).mew_y)/model.(model_name).sigma_y;

        % to remove wavelets that do not change
        index_x=var(Wmat_train.(wav_case{wav_index}))>eps;

        [B,FitInfo] = lasso(zX(:,index_x), zY,'Standardize',false,'CV',cvp,'Lambda',lasso_lambda);
        model.(model_name).FitInfo=FitInfo;
        model.(model_name).index_x=index_x;

        figure
        lassoPlot(B,FitInfo,PlotType="CV");
        legend("show")

        idxLambdaMinMSE = FitInfo.IndexMinMSE;
        % minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0);
        model.(model_name).LambdaMinMSE=FitInfo.LambdaMinMSE;
        b=zeros(size(zX,2),1);
        b(index_x)=B(:,idxLambdaMinMSE);
        model.(model_name).b=b;
        model.(model_name).b0 = FitInfo.Intercept(idxLambdaMinMSE);

        Yhat=(model.(model_name).b0+zX*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;
        [ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
        RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

        figure
        plot(Yhat,Ytrain,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        % Obtain coefficients on the original domain & check transformation
        [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});
        model.(model_name).beta0=wavFilter*diag(1./model.(model_name).sigma_x)*model.(model_name).b;
        model.(model_name).mewrec=waveletReconstruct( model.(model_name).mew_x, Wmat_bk.(wav_case{wav_index}), Wmat_wname.(wav_case{wav_index}) );

        % Yhat_check=(model.(model_name).b0+(Xtrain*wavFilter-model.(model_name).mew_x)*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;
        Yhat_check=(model.(model_name).b0+(Xtrain-model.(model_name).mew_x*wavRec)*wavFilter*diag(1./model.(model_name).sigma_x)*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;
        % Yhat_check=(model.(model_name).b0+(Xtrain-mewrec)*wavFilter*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;

        % figure
        % plot(Yhat_check,Yhat,'.')
        % XL=xlim;
        % hold on
        % plot(XL,XL,'-k')
        % hold off
        % xlabel('y_{hat-check}')
        % ylabel('y_{hat}')
        % title(['Check-',replace(model_name,'_','-')])
        % set(gca,'FontSize',16)

        figure
        stairs(lambda,model.(model_name).beta0)
        xlabel(lambda_label)
        ylabel('b')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

    end
end

%--------------------------------------------------------------------------

% Forward-PLS -------------------------------------------------------------

model_name='FORWARD_PLS';

mew_x=mean(Xtrain);
zX=Xtrain-mew_x;

mew_y=mean(Ytrain);
sigma_y=std(Ytrain);
zY=(Ytrain-mew_y)/sigma_y;

nInter=30;
scaling_method='mean-centering';

[model.(model_name)] = iplsForwardModel(zX, zY, rndIndex, nInter,scaling_method);
model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
model.(model_name).mew_x=mew_x;
model.(model_name).mew_y=mew_y;
model.(model_name).sigma_y=sigma_y;

Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
[ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

figure
plot(Yhat,Ytrain,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

b_aux=zeros(size(zX,2),1);
b_aux(model.(model_name).inModel)=model.(model_name).BETA.Y;
model.(model_name).beta0=b_aux;

figure
stairs(lambda,model.(model_name).beta0)
xlabel(lambda_label)
ylabel('b')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

% Forward-wavelet-PLS -----------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,

        model_name=['FORWARD_WAV_PLS_',scaling_option{s},'_',(wav_case{wav_index})];

        switch scaling_option{s},
            case 'MC'
                mew_x=mean(Wmat_train.(wav_case{wav_index}));
                sigma_x=ones(1,size(Wmat_train.(wav_case{wav_index}),2));
            case 'AS'
                mew_x=mean(Wmat_train.(wav_case{wav_index}));
                sigma_x=std(Wmat_train.(wav_case{wav_index}));
        end

        zX=zScale(Wmat_train.(wav_case{wav_index}),mew_x,sigma_x);

        mew_y=mean(Ytrain);
        sigma_y=std(Ytrain);
        zY=(Ytrain-mew_y)/sigma_y;

        % to remove wavelets that do not change
        index_x=var(Wmat_train.(wav_case{wav_index}))>eps;

        % mergefactor=40*2^Jdec;
        mergefactor=2^Jdec;

        aux=string(ceil(Wmat_coef_number_k.(wav_case{wav_index})/mergefactor)')+':'+string(Wmat_coef_number_s.(wav_case{wav_index})');
        aux(~index_x)=[];
        [u_aux]=unique(aux);
        interval_index=nan(1,length(aux));
        for i=1:length(u_aux),
            interval_index(aux==u_aux(i))=i;
        end

        [model.(model_name)] = iplsForwardModel(zX(:,index_x), zY, rndIndex, interval_index);
        inModel=false(1,size(zX,2));
        inModel(index_x)=model.(model_name).inModel;
        model.(model_name).inModel=inModel;
        model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
        model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
        model.(model_name).mew_x=mew_x;
        model.(model_name).sigma_x=sigma_x;
        model.(model_name).mew_y=mew_y;
        model.(model_name).sigma_y=sigma_y;

        Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
        [ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
        RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

        figure
        plot(Yhat,Ytrain,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        % Obtain coefficients on the original domain & check transformation
        [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});
        b=zeros(size(zX,2),1);
        b(model.(model_name).inModel)=model.(model_name).BETA.Y;
        model.(model_name).beta0=wavFilter*diag(1./model.(model_name).sigma_x)*b;
        model.(model_name).mewrec=waveletReconstruct( model.(model_name).mew_x, Wmat_bk.(wav_case{wav_index}), Wmat_wname.(wav_case{wav_index}) );

        % Yhat_check=(Xtrain*wavFilter-model.(model_name).mew_x)*b*model.(model_name).sigma_y+model.(model_name).mew_y;
        Yhat_check=(Xtrain-model.(model_name).mew_x*wavRec)*wavFilter*diag(1./model.(model_name).sigma_x)*b*model.(model_name).sigma_y+model.(model_name).mew_y;
        % Yhat_check=(Xtrain-mewrec)*wavFilter*b*model.(model_name).sigma_y+model.(model_name).mew_y;

        % figure
        % plot(Yhat_check,Yhat,'.')
        % XL=xlim;
        % hold on
        % plot(XL,XL,'-k')
        % hold off
        % xlabel('y_{hat-check}')
        % ylabel('y_{hat}')
        % title(['Check-',replace(model_name,'_','-')])
        % set(gca,'FontSize',16)

        figure
        stairs(lambda,model.(model_name).beta0)
        xlabel(lambda_label)
        ylabel('b')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

    end
end

%--------------------------------------------------------------------------

% Backwards-PLS -----------------------------------------------------------

model_name='BACKWARDS_PLS';

mew_x=mean(Xtrain);
zX=Xtrain-mew_x;

mew_y=mean(Ytrain);
sigma_y=std(Ytrain);
zY=(Ytrain-mew_y)/sigma_y;

nInter=30;
scaling_method='mean-centering';

[model.(model_name)] = iplsBackwardsModel(zX, zY, rndIndex, nInter,scaling_method);
model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
model.(model_name).mew_x=mew_x;
model.(model_name).mew_y=mew_y;
model.(model_name).sigma_y=sigma_y;

Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
[ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

figure
plot(Yhat,Ytrain,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

b_aux=zeros(size(zX,2),1);
b_aux(model.(model_name).inModel)=model.(model_name).BETA.Y;
model.(model_name).beta0=b_aux;

figure
stairs(lambda,model.(model_name).beta0)
xlabel(lambda_label)
ylabel('b')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

% Backwards-wavelet-PLS ---------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,

        model_name=['BACKWARDS_WAV_PLS_',scaling_option{s},'_',(wav_case{wav_index})];

        switch scaling_option{s},
            case 'MC'
                mew_x=mean(Wmat_train.(wav_case{wav_index}));
                sigma_x=ones(1,size(Wmat_train.(wav_case{wav_index}),2));
            case 'AS'
                mew_x=mean(Wmat_train.(wav_case{wav_index}));
                sigma_x=std(Wmat_train.(wav_case{wav_index}));
        end

        zX=zScale(Wmat_train.(wav_case{wav_index}),mew_x,sigma_x);

        mew_y=mean(Ytrain);
        sigma_y=std(Ytrain);
        zY=(Ytrain-mew_y)/sigma_y;

        % to remove wavelets that do not change
        index_x=var(Wmat_train.(wav_case{wav_index}))>eps;

        % mergefactor=40*2^Jdec;
        mergefactor=2^Jdec;

        aux=string(ceil(Wmat_coef_number_k.(wav_case{wav_index})/mergefactor)')+':'+string(Wmat_coef_number_s.(wav_case{wav_index})');
        aux(~index_x)=[];
        [u_aux]=unique(aux);
        interval_index=nan(1,length(aux));
        for i=1:length(u_aux),
            interval_index(aux==u_aux(i))=i;
        end

        [model.(model_name)] = iplsBackwardsModel(zX(:,index_x), zY, rndIndex, interval_index);
        inModel=false(1,size(zX,2));
        inModel(index_x)=model.(model_name).inModel;
        model.(model_name).inModel=inModel;
        model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
        model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
        model.(model_name).mew_x=mew_x;
        model.(model_name).sigma_x=sigma_x;
        model.(model_name).mew_y=mew_y;
        model.(model_name).sigma_y=sigma_y;

        Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
        [ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
        RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));

        figure
        plot(Yhat,Ytrain,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        % Obtain coefficients on the original domain & check transformation
        [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});
        b=zeros(size(zX,2),1);
        b(model.(model_name).inModel)=model.(model_name).BETA.Y;
        model.(model_name).beta0=wavFilter*diag(1./model.(model_name).sigma_x)*b;
        model.(model_name).mewrec=waveletReconstruct( model.(model_name).mew_x, Wmat_bk.(wav_case{wav_index}), Wmat_wname.(wav_case{wav_index}) );

        % Yhat_check=(Xtrain*wavFilter-model.(model_name).mew_x)*b*model.(model_name).sigma_y+model.(model_name).mew_y;
        Yhat_check=(Xtrain-model.(model_name).mew_x*wavRec)*wavFilter*diag(1./model.(model_name).sigma_x)*b*model.(model_name).sigma_y+model.(model_name).mew_y;
        % Yhat_check=(Xtrain-mewrec)*wavFilter*b*model.(model_name).sigma_y+model.(model_name).mew_y;

        % figure
        % plot(Yhat_check,Yhat,'.')
        % XL=xlim;
        % hold on
        % plot(XL,XL,'-k')
        % hold off
        % xlabel('y_{hat-check}')
        % ylabel('y_{hat}')
        % title(['Check-',replace(model_name,'_','-')])
        % set(gca,'FontSize',16)

        figure
        stairs(lambda,model.(model_name).beta0)
        xlabel(lambda_label)
        ylabel('b')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

    end
end

%--------------------------------------------------------------------------

% % Forward-Stepwise-PLS ----------------------------------------------------
% 
% model_name='STEPWISE_PLS';
% 
% mew_x=mean(Xtrain);
% zX=Xtrain-mew_x;
% 
% mew_y=mean(Ytrain);
% sigma_y=std(Ytrain);
% zY=(Ytrain-mew_y)/sigma_y;
% 
% nInter=30;
% scaling_method='mean-centering';
% 
% [model.(model_name)] = iplsForwardStepwiseModel(zX, zY, rndIndex, nInter,scaling_method);
% model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
% model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
% model.(model_name).mew_x=mew_x;
% model.(model_name).mew_y=mew_y;
% model.(model_name).sigma_y=sigma_y;
% 
% Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
% [ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
% RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));
% 
% figure
% plot(Yhat,Ytrain,'.','MarkerSize',12)
% XL=xlim;
% hold on
% plot(XL,XL,'-k')
% hold off
% xlabel('y_{hat}')
% ylabel('y')
% title(replace(model_name,'_','-'))
% set(gca,'FontSize',16)
% 
% b_aux=zeros(size(zX,2),1);
% b_aux(model.(model_name).inModel)=model.(model_name).BETA.Y;
% model.(model_name).beta0=b_aux;
% 
% figure
% stairs(lambda,b_aux)
% xlabel(lambda_label)
% ylabel('b')
% title(replace(model_name,'_','-'))
% set(gca,'FontSize',16)
% 
% %--------------------------------------------------------------------------
% 
% Forward-Stepwise-wavelet-PLS --------------------------------------------
% 
% for wav_index=1:5,
% 
%     model_name=['STEPWISE_WAV_PLS_',(wav_case{wav_index})];
% 
%     mew_x=mean(Wmat_train.(wav_case{wav_index}));
%     zX=Wmat_train.(wav_case{wav_index})-mew_x;
% 
%     mew_y=mean(Ytrain);
%     sigma_y=std(Ytrain);
%     zY=(Ytrain-mew_y)/sigma_y;
% 
%      % to remove wavelets that do not change
%     index_x=var(zX)>eps;
% 
%     % mergefactor=40*2^Jdec;
%     mergefactor=2^Jdec;
% 
%     aux=string(ceil(Wmat_coef_number_k.(wav_case{wav_index})/mergefactor)')+':'+string(Wmat_coef_number_s.(wav_case{wav_index})');
%     aux(~index_x)=[];
%     [u_aux]=unique(aux);
%     interval_index=nan(1,length(aux));
%     for i=1:length(u_aux),
%         interval_index(aux==u_aux(i))=i;
%     end
% 
%     [model.(model_name)] = iplsForwardStepwiseModel(zX(:,index_x), zY, rndIndex, interval_index);
%     inModel=false(1,size(zX,2));
%     inModel(index_x)=model.(model_name).inModel;
%     model.(model_name).inModel=inModel;
%     model.(model_name).BETA.T=model.(model_name).W*inv(model.(model_name).P'*model.(model_name).W);
%     model.(model_name).BETA.Y=model.(model_name).BETA.T*model.(model_name).B*model.(model_name).Q';
%     model.(model_name).mew_x=mew_x;
%     model.(model_name).mew_y=mew_y;
%     model.(model_name).sigma_y=sigma_y;
% 
%     Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
%     [ R2_train.(model_name) ] = funR2_coefDet( Ytrain, Yhat );
%     RMSE_train.(model_name)=sqrt(mean((Ytrain-Yhat).^2));
% 
%     figure
%     plot(Yhat,Ytrain,'.','MarkerSize',12)
%     XL=xlim;
%     hold on
%     plot(XL,XL,'-k')
%     hold off
%     xlabel('y_{hat}')
%     ylabel('y')
%     title(replace(model_name,'_','-'))
%     set(gca,'FontSize',16)
% 
%     % Obtain coefficients on the original domain & check transformation
%     [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});
%     b=zeros(size(zX,2),1);
%     b(model.(model_name).inModel)=model.(model_name).BETA.Y;
%     model.(model_name).beta0=wavFilter*b;
%     model.(model_name).mewrec=waveletRecunstruct( model.(model_name).mew_x, Wmat_bk.(wav_case{wav_index}), Wmat_wname.(wav_case{wav_index}) );
% 
%     % Yhat_check=(Xtrain*wavFilter-model.(model_name).mew_x)*b*model.(model_name).sigma_y+model.(model_name).mew_y;
%     Yhat_check=(Xtrain-model.(model_name).mew_x*wavRec)*wavFilter*b*model.(model_name).sigma_y+model.(model_name).mew_y;
%     % Yhat_check=(Xtrain-mewrec)*wavFilter*b*model.(model_name).sigma_y+model.(model_name).mew_y;
% 
%     figure
%     plot(Yhat_check,Yhat,'.')
%     XL=xlim;
%     hold on
%     plot(XL,XL,'-k')
%     hold off
%     xlabel('y_{hat-check}')
%     ylabel('y_{hat}')
%     title(['Check-',replace(model_name,'_','-')])
%     set(gca,'FontSize',16)
% 
%     figure
%     stairs(lambda,model.(model_name).beta0)
%     xlabel(lambda_label)
%     ylabel('b')
%     title(replace(model_name,'_','-'))
%     set(gca,'FontSize',16)
% 
% end

%--------------------------------------------------------------------------

% Plots results -----------------------------------------------------------

figure
model_names=fieldnames(R2_train);
x=nan(length(model_names),1);
for i=1:length(model_names),
x(i)=R2_train.(model_names{i});
end
bar(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ylabel('R^2 (train)')
set(gca,'FontSize',16)

figure
model_names=fieldnames(RMSE_train);
x=nan(length(model_names),1);
for i=1:length(model_names),
x(i)=RMSE_train.(model_names{i});
end
bar(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ub=sqrt(mean((Ytrain-mean(Ytrain)).^2));
hold on
yline(ub,'-r')
hold off
ylabel('RMSE (train)')
set(gca,'FontSize',16)

%==========================================================================

%% Test - prediction ======================================================

% Standard-PLS ------------------------------------------------------------

model_name='STD_PLS';

zX=Xtest-model.(model_name).mew_x;
Yhat=zX*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;

figure
plot(Yhat,Ytest,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
[ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

% RMSE_test_org.(model_name)=sqrt(mean((Ytest_org-exp(Yhat)).^2));
% [ R2_test_org.(model_name) ] = funR2_coefDet( Ytest_org, exp(Yhat) );

%--------------------------------------------------------------------------

% Standard-LASSO ----------------------------------------------------------

model_name='STD_LASSO';

zX=Xtest-model.(model_name).mew_x;

Yhat=(model.(model_name).b0+zX*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;

figure
plot(Yhat,Ytest,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
[ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

%--------------------------------------------------------------------------

% Wavelet-LASSO -----------------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,


        model_name=['WAV_LASSO_',scaling_option{s},'_',(wav_case{wav_index})];

        zX=zScale(Wmat_test.(wav_case{wav_index}),model.(model_name).mew_x,model.(model_name).sigma_x);

        Yhat=(model.(model_name).b0+zX*model.(model_name).b)*model.(model_name).sigma_y+model.(model_name).mew_y;

        figure
        plot(Yhat,Ytest,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
        [ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

    end
end

%--------------------------------------------------------------------------

% Forward-PLS -------------------------------------------------------------

model_name='FORWARD_PLS';

zX=Xtest-model.(model_name).mew_x;
Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;

figure
plot(Yhat,Ytest,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
[ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

%--------------------------------------------------------------------------

% Forward-Wavelet-PLS -----------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,

        model_name=['FORWARD_WAV_PLS_',scaling_option{s},'_',(wav_case{wav_index})];

        zX=zScale(Wmat_test.(wav_case{wav_index}),model.(model_name).mew_x,model.(model_name).sigma_x);

        Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;

        figure
        plot(Yhat,Ytest,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
        [ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

    end
end

%--------------------------------------------------------------------------

% Backwards-PLS -------------------------------------------------------------

model_name='BACKWARDS_PLS';

zX=Xtest-model.(model_name).mew_x;
Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;

figure
plot(Yhat,Ytest,'.','MarkerSize',12)
XL=xlim;
hold on
plot(XL,XL,'-k')
hold off
xlabel('y_{hat}')
ylabel('y')
title(replace(model_name,'_','-'))
set(gca,'FontSize',16)

RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
[ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

%--------------------------------------------------------------------------

% Backwards-Wavelet-PLS ---------------------------------------------------

scaling_option={'MC','AS'};

for s=1:length(scaling_option),
    for wav_index=1:5,

        model_name=['BACKWARDS_WAV_PLS_',scaling_option{s},'_',(wav_case{wav_index})];

        zX=zScale(Wmat_test.(wav_case{wav_index}),model.(model_name).mew_x,model.(model_name).sigma_x);

        Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;

        figure
        plot(Yhat,Ytest,'.','MarkerSize',12)
        XL=xlim;
        hold on
        plot(XL,XL,'-k')
        hold off
        xlabel('y_{hat}')
        ylabel('y')
        title(replace(model_name,'_','-'))
        set(gca,'FontSize',16)

        RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
        [ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );

    end
end

%--------------------------------------------------------------------------

% % Forward-Stepwise-PLS ----------------------------------------------------
% 
% model_name='STEPWISE_PLS';
% 
% zX=Xtest-model.(model_name).mew_x;
% Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
% 
% figure
% plot(Yhat,Ytest,'.','MarkerSize',12)
% XL=xlim;
% hold on
% plot(XL,XL,'-k')
% hold off
% xlabel('y_{hat}')
% ylabel('y')
% title(replace(model_name,'_','-'))
% set(gca,'FontSize',16)
% 
% RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
% [ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );
% 
% %--------------------------------------------------------------------------
% 
% % Forward-Stepwise-Wavelet-PLS --------------------------------------------
% 
% for wav_index=1:5,
% 
%     model_name=['STEPWISE_WAV_PLS_',(wav_case{wav_index})];
% 
%     zX=Wmat_test.(wav_case{wav_index})-model.(model_name).mew_x;
%     Yhat=zX(:,model.(model_name).inModel)*model.(model_name).BETA.Y*model.(model_name).sigma_y+model.(model_name).mew_y;
% 
%     figure
%     plot(Yhat,Ytest,'.','MarkerSize',12)
%     XL=xlim;
%     hold on
%     plot(XL,XL,'-k')
%     hold off
%     xlabel('y_{hat}')
%     ylabel('y')
%     title(replace(model_name,'_','-'))
%     set(gca,'FontSize',16)
% 
%     RMSE_test.(model_name)=sqrt(mean((Ytest-Yhat).^2));
%     [ R2_test.(model_name) ] = funR2_coefDet( Ytest, Yhat );
% 
% end
% 
% %--------------------------------------------------------------------------

% Plot results ------------------------------------------------------------

figure
model_names=fieldnames(R2_test);
x=nan(length(model_names),1);
for i=1:length(model_names),
x(i)=R2_test.(model_names{i});
end
bar(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ylabel('R^2 (test)')
set(gca,'FontSize',16)

figure
model_names=fieldnames(RMSE_test);
x=nan(length(model_names),1);
for i=1:length(model_names),
x(i)=RMSE_test.(model_names{i});
end
bar(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ub=sqrt(mean((Ytest-mean(Ytest)).^2));
hold on
yline(ub,'-r')
hold off
ylabel('RMSE (test)')
set(gca,'FontSize',16)

filename='beer_stage_1.mat';
save(fullfile(resultsFolder,'lasso_wavelets',filename),'model','RMSE_train','R2_train','RMSE_test','R2_test','rndIndex','cvp')

%==========================================================================