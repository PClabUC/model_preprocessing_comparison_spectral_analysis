function [ Xrec ] = waveletReconstruct( W, L, wname )
% INPUTS:
% W     : Matrix of wavelet coefficients
% L     : Bookkeeping vector
% wname : Name of Wavelet Transform
%
% OUTPUTS:
% Xrec : Reconstructed signal


n=size(W,1);

Xrec=nan(n,L(end));
for i=1:n,
Xrec(i,:) = waverec(W(i,:),L,wname);
end

end

