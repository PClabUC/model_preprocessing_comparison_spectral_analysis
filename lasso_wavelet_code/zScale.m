function [ Z ] = zScale( X, mew, sigma )
% ZSCALE scales the original data based on a given mean and standard deviation.
%
% Required input arguments:
%   X     : A data matrix of observations where rows represent observations, and columns represent variables. 
%   mew   : A vector with the mean value of each variable.
%   sigma : A vector with the standard deviation of each variable.
%
% I/O: [ Z ] = zScale( X, mew, sigma )
%
% The output of ZSCALE is:
%   Z : A scaled data matrix.
 
% data size
% [n,~,r]=size(X);
 
% scaling
% Z=(X-repmat(mew,[n,1,r]))./repmat(sigma,[n,1,r]);
 
Z=(X-mew)./sigma;

end