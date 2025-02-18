clear
clc
close all

dataFolder = fullfile(cd, '..','Data');
resultsFolder = fullfile(cd, '..','Results');

%% Load data ==============================================================

% Load data ---------------------------------------------------------------

Xraw=readmatrix(fullfile(dataFolder, 'beer_spectra.csv'));
Xtrain=Xraw(:,1:end-1);
Ytrain=Xraw(:,end);

Xraw=readmatrix(fullfile(dataFolder, 'beer_spectra_test.csv'));
Xtest=Xraw(:,1:end-1);
Ytest=Xraw(:,end);

lambda=readmatrix(fullfile(dataFolder, 'wavelenghts.csv'));
lambda_label='\tilde{nu} (cm^{-1})';

clear Xraw

% -------------------------------------------------------------------------

% Load models -------------------------------------------------------------

filename='beer_stage_1.mat';
load(fullfile(resultsFolder,'lasso_wavelets',filename),'model')

% -------------------------------------------------------------------------

% Load random datasets for Stage 2 of SS-DAC ------------------------------

index_Q=readmatrix(fullfile(dataFolder, 'ss-dac-test-id-Q.csv'));
index_R=readmatrix(fullfile(dataFolder, 'ss-dac-test-id-R.csv'));

Q=size(index_Q,1);
R=size(index_R,1);

% -------------------------------------------------------------------------

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


%% Retrain models =========================================================

% standard-PLS ------------------------------------------------------------

model_name='STD_PLS';

% hiper-parameters
kpls=model.(model_name).k;

R2_train_q.(model_name)=nan(Q,1);
RMSE_train_q.(model_name)=nan(Q,1);
model_Q.(model_name)=cell(Q,1);

for q=1:Q,

    Xtrain_q=Xtrain(index_Q(q,:),:);
    Ytrain_q=Ytrain(index_Q(q,:));

    model_q.(model_name).mew_x=mean(Xtrain_q);
    zX=Xtrain_q-model_q.(model_name).mew_x;

    model_q.(model_name).mew_y=mean(Ytrain_q);
    model_q.(model_name).sigma_y=std(Ytrain_q);
    zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

    [ model_q.(model_name).P, model_q.(model_name).Q, model_q.(model_name).B, model_q.(model_name).W, model_q.(model_name).BETA ] = plsModel( zX, zY, kpls );
    model_q.(model_name).beta0=model_q.(model_name).BETA.Y;
    model_q.(model_name).k=kpls;

    Yhat=zX*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
    R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
    RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

    model_Q.(model_name){q}=model_q;
end

%--------------------------------------------------------------------------

% Standard LASSO ----------------------------------------------------------

model_name='STD_LASSO';

% hyper-parameters
LambdaMinMSE=model.(model_name).LambdaMinMSE;

R2_train_q.(model_name)=nan(Q,1);
RMSE_train_q.(model_name)=nan(Q,1);
model_Q.(model_name)=cell(Q,1);

for q=1:Q,

    Xtrain_q=Xtrain(index_Q(q,:),:);
    Ytrain_q=Ytrain(index_Q(q,:));

    model_q.(model_name).mew_x=mean(Xtrain_q);
    zX=Xtrain_q-model_q.(model_name).mew_x;

    model_q.(model_name).mew_y=mean(Ytrain_q);
    model_q.(model_name).sigma_y=std(Ytrain_q);
    zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

    [B,FitInfo] = lasso(zX, zY,'Standardize',false,'Lambda',LambdaMinMSE);%,'Intercept',false);
    model_q.(model_name).FitInfo=FitInfo;

    model_q.(model_name).LambdaMinMSE=LambdaMinMSE;
    model_q.(model_name).b=B;
    model_q.(model_name).b0 = FitInfo.Intercept;
    model_q.(model_name).beta0=model_q.(model_name).b;

    Yhat=(model_q.(model_name).b0+zX*model_q.(model_name).b)*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
    R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
    RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

    model_Q.(model_name){q}=model_q;
end

%--------------------------------------------------------------------------

% Wavelet-LASSO -----------------------------------------------------------

for wav_index=1:5,

    model_name=['WAV_LASSO_',(wav_case{wav_index})];

    % wavelet filters
    [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});

    % hyper-parameters
    LambdaMinMSE=model.(model_name).LambdaMinMSE;
    index_x=model.(model_name).index_x;

    R2_train_q.(model_name)=nan(Q,1);
    RMSE_train_q.(model_name)=nan(Q,1);
    model_Q.(model_name)=cell(Q,1);

    for q=1:Q,

        Wtrain_q=Wmat_train.(wav_case{wav_index});
        Wtrain_q=Wtrain_q(index_Q(q,:),:);
        Ytrain_q=Ytrain(index_Q(q,:));

        model_q.(model_name).mew_x=mean(Wtrain_q);
        zX=Wtrain_q-model_q.(model_name).mew_x;

        model_q.(model_name).mew_y=mean(Ytrain_q);
        model_q.(model_name).sigma_y=std(Ytrain_q);
        zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

        [B,FitInfo] = lasso(zX(:,index_x), zY,'Standardize',false,'Lambda',LambdaMinMSE);%,'Intercept',false);
        model_q.(model_name).FitInfo=FitInfo;

        model_q.(model_name).LambdaMinMSE=LambdaMinMSE;
        b=zeros(size(zX,2),1);
        b(index_x)=B;
        model_q.(model_name).b=b;
        model_q.(model_name).b0 = FitInfo.Intercept;
        model_q.(model_name).beta0=wavFilter*model_q.(model_name).b;
        model_q.(model_name).index_x=index_x;

        Yhat=(model_q.(model_name).b0+zX*model_q.(model_name).b)*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
        R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
        RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

        model_Q.(model_name){q}=model_q;
    end
end

%--------------------------------------------------------------------------

% Forward-PLS -------------------------------------------------------------

model_name='FORWARD_PLS';

% hiper-parameters
kpls=model.(model_name).k;
inModel=model.(model_name).inModel;

R2_train_q.(model_name)=nan(Q,1);
RMSE_train_q.(model_name)=nan(Q,1);
model_Q.(model_name)=cell(Q,1);

for q=1:Q,

    Xtrain_q=Xtrain(index_Q(q,:),:);
    Ytrain_q=Ytrain(index_Q(q,:));

    model_q.(model_name).mew_x=mean(Xtrain_q);
    zX=Xtrain_q-model_q.(model_name).mew_x;

    model_q.(model_name).mew_y=mean(Ytrain_q);
    model_q.(model_name).sigma_y=std(Ytrain_q);
    zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

    [ model_q.(model_name).P, model_q.(model_name).Q, model_q.(model_name).B, model_q.(model_name).W, model_q.(model_name).BETA ] = plsModel( zX(:,inModel), zY, kpls );
    model_q.(model_name).beta0=model_q.(model_name).BETA.Y;
    model_q.(model_name).inModel=inModel;
    model_q.(model_name).k=kpls;

    Yhat=zX(:,inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
    R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
    RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

    model_Q.(model_name){q}=model_q;
end

%--------------------------------------------------------------------------

% Forward-wavelet-PLS -----------------------------------------------------

for wav_index=1:5,

    model_name=['FORWARD_WAV_PLS_',(wav_case{wav_index})];

    % wavelet filters
    [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});

    % hyper-parameters
    kpls=model.(model_name).k;
    inModel=model.(model_name).inModel;
    % index_x=model.(model_name).index_x;

    R2_train_q.(model_name)=nan(Q,1);
    RMSE_train_q.(model_name)=nan(Q,1);
    model_Q.(model_name)=cell(Q,1);

    for q=1:Q,

        Waux=Wmat_train.(wav_case{wav_index});
        Wtrain_q=Waux(index_Q(q,:),:);
        Ytrain_q=Ytrain(index_Q(q,:));

        model_q.(model_name).mew_x=mean(Wtrain_q);
        zX=Wtrain_q-model_q.(model_name).mew_x;

        model_q.(model_name).mew_y=mean(Ytrain_q);
        model_q.(model_name).sigma_y=std(Ytrain_q);
        zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

        [ model_q.(model_name).P, model_q.(model_name).Q, model_q.(model_name).B, model_q.(model_name).W, model_q.(model_name).BETA ] = plsModel( zX(:,inModel), zY, kpls );
        model_q.(model_name).inModel=inModel;
        model_q.(model_name).k=kpls;
        
        b=zeros(size(zX,2),1);
        b(model_q.(model_name).inModel)=model_q.(model_name).BETA.Y;
        model_q.(model_name).beta0=wavFilter*b;

        Yhat=zX(:,inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
        R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
        RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

        model_Q.(model_name){q}=model_q;
    end
end

%--------------------------------------------------------------------------

% Backwards-PLS -----------------------------------------------------------

model_name='BACKWARDS_PLS';

% hiper-parameters
kpls=model.(model_name).k;
inModel=model.(model_name).inModel;

R2_train_q.(model_name)=nan(Q,1);
RMSE_train_q.(model_name)=nan(Q,1);
model_Q.(model_name)=cell(Q,1);

for q=1:Q,

    Xtrain_q=Xtrain(index_Q(q,:),:);
    Ytrain_q=Ytrain(index_Q(q,:));

    model_q.(model_name).mew_x=mean(Xtrain_q);
    zX=Xtrain_q-model_q.(model_name).mew_x;

    model_q.(model_name).mew_y=mean(Ytrain_q);
    model_q.(model_name).sigma_y=std(Ytrain_q);
    zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

    [ model_q.(model_name).P, model_q.(model_name).Q, model_q.(model_name).B, model_q.(model_name).W, model_q.(model_name).BETA ] = plsModel( zX(:,inModel), zY, kpls );
    model_q.(model_name).beta0=model_q.(model_name).BETA.Y;
    model_q.(model_name).inModel=inModel;
    model_q.(model_name).k=kpls;

    Yhat=zX(:,inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
    R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
    RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

    model_Q.(model_name){q}=model_q;
end

%--------------------------------------------------------------------------

% Backwards-wavelet-PLS ---------------------------------------------------

for wav_index=1:5,

    model_name=['BACKWARDS_WAV_PLS_',(wav_case{wav_index})];

    % wavelet filters
    [wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});

    % hyper-parameters
    kpls=model.(model_name).k;
    inModel=model.(model_name).inModel;
    % index_x=model.(model_name).index_x;

    R2_train_q.(model_name)=nan(Q,1);
    RMSE_train_q.(model_name)=nan(Q,1);
    model_Q.(model_name)=cell(Q,1);

    for q=1:Q,

        Waux=Wmat_train.(wav_case{wav_index});
        Wtrain_q=Waux(index_Q(q,:),:);
        Ytrain_q=Ytrain(index_Q(q,:));

        model_q.(model_name).mew_x=mean(Wtrain_q);
        zX=Wtrain_q-model_q.(model_name).mew_x;

        model_q.(model_name).mew_y=mean(Ytrain_q);
        model_q.(model_name).sigma_y=std(Ytrain_q);
        zY=(Ytrain_q-model_q.(model_name).mew_y)/model_q.(model_name).sigma_y;

        [ model_q.(model_name).P, model_q.(model_name).Q, model_q.(model_name).B, model_q.(model_name).W, model_q.(model_name).BETA ] = plsModel( zX(:,inModel), zY, kpls );
        model_q.(model_name).inModel=inModel;
        model_q.(model_name).k=kpls;

        b=zeros(size(zX,2),1);
        b(model_q.(model_name).inModel)=model_q.(model_name).BETA.Y;
        model_q.(model_name).beta0=wavFilter*b;

        Yhat=zX(:,inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;
        R2_train_q.(model_name)(q)=funR2_coefDet( Ytrain_q, Yhat );
        RMSE_train_q.(model_name)(q)=sqrt(mean((Ytrain_q-Yhat).^2));

        model_Q.(model_name){q}=model_q;
    end
end

%--------------------------------------------------------------------------

% Plot results ------------------------------------------------------------

figure
model_names=fieldnames(R2_train_q);
x=nan(Q,length(model_names),1);
for i=1:length(model_names),
x(:,i)=R2_train_q.(model_names{i});
end
boxplot(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ylabel('R^2 (train)')
set(gca,'FontSize',16)

figure
model_names=fieldnames(RMSE_train_q);
x=nan(Q,length(model_names),1);
for i=1:length(model_names),
x(:,i)=RMSE_train_q.(model_names{i});
end
boxplot(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ub=sqrt(mean((Ytrain-mean(Ytrain)).^2));
hold on
yline(ub,'-r')
hold off
ylabel('RMSE (train)')
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

%==========================================================================

%% Test - prediction ======================================================


% Standard-PLS ------------------------------------------------------------

model_name='STD_PLS';

R2_test_r.(model_name)=nan(Q,R);
RMSE_test_r.(model_name)=nan(Q,R);

for r=1:R,

    Xtest_r=Xtest(index_R(r,:),:);
    Ytest_r=Ytest(index_R(r,:));
    for q=1:Q,

        model_q=model_Q.(model_name){q};

        zX=Xtest_r-model_q.(model_name).mew_x;
        Yhat=zX*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;


        RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
        R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

    end
end

%--------------------------------------------------------------------------

% Standard-LASSO ----------------------------------------------------------

model_name='STD_LASSO';

R2_test_r.(model_name)=nan(Q,R);
RMSE_test_r.(model_name)=nan(Q,R);

for r=1:R,

    Xtest_r=Xtest(index_R(r,:),:);
    Ytest_r=Ytest(index_R(r,:));
    for q=1:Q,

        model_q=model_Q.(model_name){q};

        zX=Xtest_r-model_q.(model_name).mew_x;
        Yhat=(model_q.(model_name).b0+zX*model_q.(model_name).b)*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

        RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
        R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

    end
end

%--------------------------------------------------------------------------

% Wavelet-LASSO -----------------------------------------------------------

for wav_index=1:5,

    model_name=['WAV_LASSO_',(wav_case{wav_index})];

    R2_test_r.(model_name)=nan(Q,R);
    RMSE_test_r.(model_name)=nan(Q,R);

    for r=1:R,

        Waux=Wmat_test.(wav_case{wav_index});
        Wtest_r=Waux(index_R(r,:),:);
        Ytest_r=Ytest(index_R(r,:));

        for q=1:Q,

            model_q=model_Q.(model_name){q};

            zX=Wtest_r-model_q.(model_name).mew_x;
            Yhat=(model_q.(model_name).b0+zX*model_q.(model_name).b)*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

            RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
            R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

        end
    end

end

%--------------------------------------------------------------------------

% Forward-PLS -------------------------------------------------------------

model_name='FORWARD_PLS';

R2_test_r.(model_name)=nan(Q,R);
RMSE_test_r.(model_name)=nan(Q,R);

for r=1:R,

    Xtest_r=Xtest(index_R(r,:),:);
    Ytest_r=Ytest(index_R(r,:));

    for q=1:Q,

        model_q=model_Q.(model_name){q};

        zX=Xtest_r-model_q.(model_name).mew_x;
        Yhat=zX(:,model_q.(model_name).inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

        RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
        R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

    end
end

%--------------------------------------------------------------------------

% Forward-Wavelet-PLS -----------------------------------------------------

for wav_index=1:5,

    model_name=['FORWARD_WAV_PLS_',(wav_case{wav_index})];

    R2_test_r.(model_name)=nan(Q,R);
    RMSE_test_r.(model_name)=nan(Q,R);

    for r=1:R,

        Waux=Wmat_test.(wav_case{wav_index});
        Wtest_r=Waux(index_R(r,:),:);
        Ytest_r=Ytest(index_R(r,:));


        for q=1:Q,

            model_q=model_Q.(model_name){q};

            zX=Wtest_r-model_q.(model_name).mew_x;
            Yhat=zX(:,model_q.(model_name).inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

            RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
            R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

        end
    end
end

%--------------------------------------------------------------------------

% Backwards-PLS ------------------------------------------------------------

model_name='BACKWARDS_PLS';

R2_test_r.(model_name)=nan(Q,R);
RMSE_test_r.(model_name)=nan(Q,R);

for r=1:R,

    Xtest_r=Xtest(index_R(r,:),:);
    Ytest_r=Ytest(index_R(r,:));

    for q=1:Q,

        model_q=model_Q.(model_name){q};

        zX=Xtest_r-model_q.(model_name).mew_x;
        Yhat=zX(:,model_q.(model_name).inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

        RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
        R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

    end
end

%--------------------------------------------------------------------------

% Backwards-Wavelet-PLS ---------------------------------------------------

for wav_index=1:5,

    model_name=['BACKWARDS_WAV_PLS_',(wav_case{wav_index})];

    R2_test_r.(model_name)=nan(Q,R);
    RMSE_test_r.(model_name)=nan(Q,R);

    for r=1:R,

        Waux=Wmat_test.(wav_case{wav_index});
        Wtest_r=Waux(index_R(r,:),:);
        Ytest_r=Ytest(index_R(r,:));


        for q=1:Q,

            model_q=model_Q.(model_name){q};

            zX=Wtest_r-model_q.(model_name).mew_x;
            Yhat=zX(:,model_q.(model_name).inModel)*model_q.(model_name).BETA.Y*model_q.(model_name).sigma_y+model_q.(model_name).mew_y;

            RMSE_test_r.(model_name)(q,r)=sqrt(mean((Ytest_r-Yhat).^2));
            R2_test_r.(model_name)(q,r) = funR2_coefDet( Ytest_r, Yhat );

        end
    end
end

%--------------------------------------------------------------------------

figure
model_names=fieldnames(R2_test_r);
x=nan(Q*R,length(model_names));
for i=1:length(model_names),
    xaux=R2_test_r.(model_names{i});
    x(:,i)=xaux(:);
end
boxplot(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
ylabel('R^2 (test)')
set(gca,'FontSize',16)

figure
model_names=fieldnames(RMSE_test_r);
x=nan(Q*R,length(model_names));
for i=1:length(model_names),
    xaux=RMSE_test_r.(model_names{i});
    x(:,i)=xaux(:);
end
boxplot(x)
xticks(1:length(model_names))
xticklabels(replace(model_names,'_','-'))
% ub=sqrt(mean((Ytest-mean(Ytest)).^2));
% hold on
% yline(ub,'-r')
% hold off
ylabel('RMSE (test)')
set(gca,'FontSize',16)

index_1=7;
index_2=13;
figure
hold on
for r=1:R,
    plot(RMSE_test_r.(model_names{index_1})(:,r),RMSE_test_r.(model_names{index_2})(:,r),'.')
end
clim=funDiag();
xlim(clim);
ylim(clim);
hold off
box on
xlabel(replace(model_names{index_1},'_','-'))
ylabel(replace(model_names{index_2},'_','-'))
set(gca,'FontSize',16)

filename='beer_stage_2.mat';
save(fullfile(resultsFolder,'lasso_wavelets',filename),'R2_train_q','RMSE_test_r','R2_test_r')

% export KPI (RMSE) -------------------------------------------------------

models_to_export={
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

for i=1:length(models_to_export),
    filename=['kpi_',models_to_export{i},'.csv'];
    writematrix(RMSE_test_r.(models_to_export{i}),fullfile(resultsFolder,'lasso_wavelets','model_kpi',filename))
end

%--------------------------------------------------------------------------

%==========================================================================