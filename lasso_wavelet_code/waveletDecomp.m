function [ W, L, w_coef_number_k, w_coef_number_s ] = waveletDecomp( X, Jdec, wname )
% INPUTS:
% X     : (n x m) data matrix, where n is the number of samples and m is the
% number of measurements
% Jdec  : Decomposition depth
% wname : Name of Wavelet Transform
%
% OUTPUTS:
% W               : Matrix of wavelet coefficients
% L               : Bookkeeping vector
% w_coef_number_k : window indices
% w_coef_number_s : scale indices

[n,m]=size(X);

[w,L] = wavedec(X(1,:),Jdec,wname);
mW=sum(L(1:end-1));
W=nan(n,mW);
W(1,:)=w;
for i=2:n,
    [W(i,:)] = wavedec(X(i,:),Jdec,wname);
end

w_coef_number_s=nan(1,mW);
ind_end=0;
for i=1:Jdec+1,
    ind_start=ind_end+1;
    ind_end=ind_start+L(i)-1;
    w_coef_number_s(ind_start:ind_end)=Jdec+2-i;
end

w_coef_number_k=nan(1,mW);
ind_end=0;
for i=1:Jdec+1,
    if i==1,
        J=Jdec;
    else
        J=Jdec+2-i;
    end
    ind_start=ind_end+1;
    ind_end=ind_start+L(i)-1;
   w_coef_number_k(ind_start:ind_end)=2^J:2^J:L(i)*2^J;
end


end

