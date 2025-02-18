function [ R2 ] = funR2_coefDet( y_meas, y_pred )
% 19.06.2020


SQtot=nanmean((y_meas-nanmean(y_meas)).^2);
SQres=nanmean((y_meas-y_pred).^2);
R2=1-SQres/SQtot;

end

