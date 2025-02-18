function [VIP] = plsVip(X,W,betaT,b)
% Computes VIP of PLS model.

[M,K]=size(W);

T=X*betaT;

St=T'*T;
SSY_k=b.^2.*diag(St);
SSY_tot=sum(SSY_k);

VIP=nan(M,1);
for i=1:M,
    VIP(i)=sqrt(M*sum(W(i,:).^2*SSY_k/SSY_tot));
end


end

