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

clear Xraw

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

    filename=[dataFolder,'/wavelet_coefficients/wavelet_',wname{wav_index},'_train.csv'];
    writematrix(W,filename);

    figure
    subplot(2,3,1)
    plot(Xrec')
    for r=2:Jdec+1,
        Wrec=W;
        Wrec(:,w_coef_number_s<r)=0;
        [ Xrec ] = waveletReconstruct( Wrec, wavBK, wname{wav_index} );

        filename=[dataFolder,'/spectra_reconstructions/',wav_case{wav_index},'/rec_spectra_',wname{wav_index},'_',num2str(r),'_train.csv'];
        writematrix(Xrec,filename);

        subplot(2,3,r)
        plot(Xrec')
    end

end

%--------------------------------------------------------------------------

% test dataset ------------------------------------------------------------

for wav_index=1:5,

    [ W, wavBK, w_coef_number_k, w_coef_number_s ] = waveletDecomp( Xtest, Jdec, wname{wav_index} );
    % [ Xrec ] = waveletReconstruct( W, wavBK, wname{wav_index} );

    filename=[dataFolder,'/wavelet_coefficients/wavelet_',wname{wav_index},'_test.csv'];
    writematrix(W,filename);

    for r=2:Jdec+1,
        Wrec=W;
        Wrec(:,w_coef_number_s<r)=0;
        [ Xrec ] = waveletReconstruct( Wrec, wavBK, wname{wav_index} );

        filename=[dataFolder,'/spectra_reconstructions/',wav_case{wav_index},'/rec_spectra_',wname{wav_index},'_',num2str(r),'_test.csv'];
        writematrix(Xrec,filename);
    end
end

%--------------------------------------------------------------------------

%==========================================================================

