function [CL] = funDiag(dummy)
XL=xlim;
YL=ylim;
CL=[min(XL(1),YL(1)) max(XL(2),YL(2))];

plot(CL,CL,'-k')
end