%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [A] = PCA(X, num_PCs)
% Calculate the Transform matrix of PCA
% input:
% X         :   data matrix to be transformed 
% num_PCs   :   number of principle compoents needed
%
% ouput:
% A         :   Transform matrix in size: d by num_PCs              

X = double(X);
X = transpose(X);

[d N] = size(X);
mu = mean(X, 2);
rep_mu = repmat(mu,1, N);

centred_mtx = X - rep_mu;

[U S V] = svd(centred_mtx);

A = U(:, 1:num_PCs);

end