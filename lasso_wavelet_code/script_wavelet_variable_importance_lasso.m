clear
clc
close all

cmap=colormap('lines'); close all;

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
lambda_lim=[min(lambda) max(lambda)];

clear Xraw

%==========================================================================

% Load models -------------------------------------------------------------

filename='beer_stage_1.mat';
load(fullfile(resultsFolder,'lasso_wavelets',filename),'model','RMSE_train','R2_train','RMSE_test','R2_test','rndIndex')

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

%% Represent model parameters =============================================

% Select model ------------------------------------------------------------

% model_name='WAV_LASSO_Haar'; wav_index=1;
model_name='WAV_LASSO_Daubechies_4'; wav_index=2;
% model_name='WAV_LASSO_Daubechies_6'; wav_index=3;
% model_name='WAV_LASSO_Symmlet_4'; wav_index=4;  
% model_name='WAV_LASSO_Symmlet_6'; wav_index=5;                     
 
%--------------------------------------------------------------------------

% Coefficients ------------------------------------------------------------

% wavelet filters
[wavFilter, wavRec] = waveletFilter(size(Xtrain,2),Jdec,wname{wav_index});
m=size(wavFilter,2);

b=model.(model_name).b;

w_coef_number_s=Wmat_coef_number_s.(wav_case{wav_index});

% contributions by wavelet coefficient
betaByWavCoef=(wavFilter~=0)*diag(b);
betaOriginalDomain=wavFilter*diag(b);

% coefficients by wavelet
figure
hold on
for s=1:Jdec+1,
    plot(nan,nan,'color',cmap(s,:))
end
for i=1:m,
    stairs(lambda,betaByWavCoef(:,i),'color',cmap(w_coef_number_s(i),:),'HandleVisibility', 'off')
end
yline(0,'k-')
hold off
xlabel(lambda_label)
ylabel('b')
xlim(lambda_lim)
title(replace(model_name,'_','-'))
legend(["d_"+num2str((1:Jdec)');"a_"+num2str(Jdec)])
box on
set(gca,'FontSize',16)

% coefficients by wavelet in original domain (accounting by wavelet filter)
figure
hold on
for s=1:Jdec+1,
    plot(nan,nan,'color',cmap(s,:))
end
for i=1:m,
    stairs(lambda,betaOriginalDomain(:,i),'color',cmap(w_coef_number_s(i),:),'HandleVisibility', 'off')
end
yline(0,'k-')
hold off
xlabel(lambda_label)
ylabel('b')
xlim(lambda_lim)
title(replace(model_name,'_','-'))
legend(["d_"+num2str((1:Jdec)');"a_"+num2str(Jdec)])
box on
set(gca,'FontSize',16)

% coefficients by wavelet in original domain
figure
hold on
for s=1:Jdec+1,
    plot(nan,nan,'color',cmap(s,:))
end
stairs(lambda,sum(betaOriginalDomain,2),'color',cmap(w_coef_number_s(i),:),'HandleVisibility', 'off')
yline(0,'k-')
hold off
xlabel(lambda_label)
ylabel('b')
xlim(lambda_lim)
title(replace(model_name,'_','-'))
box on
set(gca,'FontSize',16)

%--------------------------------------------------------------------------

% Spectra wavelet decomposition -------------------------------------------

[ Wtrain, wavBK, w_coef_number_k, w_coef_number_s ] = waveletDecomp( Xtrain, Jdec, wname{wav_index} );
zW=Wtrain-model.(model_name).mew_x;
mX=size(Xtrain,2);

nQuantiles=4;
q=[-inf quantile(Ytrain,(1:nQuantiles-1)/nQuantiles) inf ];
leg_txt=string(q(1:nQuantiles))+" < y <= "+string(q(2:nQuantiles+1));
leg_txt(1)="y <= "+string(q(2));
leg_txt(nQuantiles)=string(q(nQuantiles))+" < y";


figure
for s=1:Jdec+1,
    subplot(Jdec+1,1,s)
    hold on

    index=w_coef_number_s==s & w_coef_number_k<=mX;% ignore wavelet coefficints at the end

    for i=1:nQuantiles,
        index_y=Ytrain>q(i) & Ytrain<=q(i+1);
        plot(lambda(w_coef_number_k(index)),zW(index_y,index),'.-','Color',cmap(i,:),'HandleVisibility', 'off')

        plot(nan,nan,'-','Color',cmap(i,:))% legend dummy
    end

    xlim(lambda_lim)
    xlabel(lambda_label)

    legend(leg_txt)

    if s<=Jdec,
        ylabel(['d_',num2str(s)])
    else
        ylabel(['a_',num2str(Jdec)])
    end

    hold off
    box on
    set(gca,'FontSize',16)
end

%--------------------------------------------------------------------------

% Reconstructed spectra ---------------------------------------------------

inModel=model.(model_name).b~=0;
zW=Wmat_train.(wav_case{wav_index});
zW(:,~inModel)=0;

% spectra reconstruction
wavelet_numCoefByLevel=Wmat_bk.(wav_case{wav_index}); % number of coefficients by level
wavelet_name=wname{wav_index};
[ Xrec ] = waveletReconstruct( zW, wavelet_numCoefByLevel, wavelet_name );

figure
plot(lambda,mean(Xrec,1))
xlim(lambda_lim)
xlabel(lambda_label)
ylabel('Reconstructed spectra')
box on
set(gca,'FontSize',16)

%--------------------------------------------------------------------------


%==========================================================================
