function [ P, Q, B, W, BETA, RES ] = plsModel( X, Y, k )
%PLSMODEL trains a PLS model, such that:
% X=T*P'+E
% Y=U*Q'+F1
% Y=T*B*Q'+F2
%
% References: 
%    Kaspar, M.H. e Ray, W.H., Chemometric Methods for Process Monitoring and
%       High-Performance Controller Design. AIChE Journal, 1992. 38(10): p. 1593-1608.
%
% Required input arguments: 
%   X      : A regressors data matrix. The rows of X represent observations, and columns represent variables.  
%   Y      : A response column vector.
%   k      : Number of latent variables.
%
% I/O: [ P, Q, B, W ] = plsModel( X, Y, k )
%
% The outputs of PLSMODEL are:
%   P    : X-loading matrix.
%   Q    : Y-loading matrix.
%   B    : Regression coefficient matrix.
%   W    : Weighting matrix on X.
%
% see also kSelectPLS
 
 
 
tolT0=1e-10;
 
[obs, nx]=size(X);
ny=size(Y,2);
B=zeros(k,1);
P=zeros(nx,k);
Q=zeros(ny,k);
W=zeros(nx,k);
T=zeros(obs,k);
itmax=20000;
tolT=sqrt(obs)*tolT0;
 
for i=1:k,
    
    iY=random('unid',ny);
    u=Y(:,iY);
    t0=ones(obs,1);
    t=zeros(obs,1);
    it=0;
    
    while norm(t0-t)>tolT && it<=itmax,
        it=it+1;
        t0=t;
        
        w=X'*u/(u'*u);
        w=w/norm(w);
        t=X*w;
        
        q=Y'*t/(t'*t);
        q=q/norm(q);
        u=Y*q;
    
    end
    
    p=X'*t/(t'*t);
    b=u'*t/(t'*t);
    
    X=X-t*p';
    Y=Y-b*t*q';
    
    P(:,i)=p;
    Q(:,i)=q;
    B(i)=b;
    W(:,i)=w;
    T(:,i)=t;
    
end
%residuals
RES.X=X;
RES.Y=Y;
   
B=diag(B);
 
if nargout>=5,% Does not compute when RES is outputed
    BETA.T=W*inv(P'*W);
    BETA.Y=BETA.T*B*Q';
else
    BETA=[];
end
 
 
end