function [wavFilter, wavRec] = waveletFilter(nx,J,wname)
% INPUTS:
% nx    : Number of measurements of the original signal
% J     : Decomposition depth
% wname : Name of Wavelet Transform
%
% OUTPUTS:
% wavFilter : Filter matrix for wavelet decomposition
% wavRec    : Filter matrix for wavelet reconstruction

% W{wavelets}=X{original}*wavFilter
% X{original}=W{wavelets}*wavRec 
% Y=W{wavelets}*b{wavelets} <=> Y=(X{original}*wavFilter)*b{wavelets}

% Wavelet filter
I=eye(nx);
[ wavFilter, L ] = waveletDecomp( I , J, wname );

% Reconstruction filter
nw=size(wavFilter,2);
I=eye(nw);
[ wavRec ] = waveletReconstruct( I , L, wname );


end

