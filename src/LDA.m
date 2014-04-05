%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [T] = LDA(X, Y, num_basis)
% Calculate the LDA bases
% input:
% X         :   data matrix to be transformed 
% Y         :   class labels
% num_bases :   number of basis needed
%
% ouput:
% T         :   basis matrix in size: d by num_basis

X = double(X);
X = transpose(X);

[d N] = size(X);

C = length(unique(Y));
prior = zeros(1, C);
each_class_size = zeros(1, C);
each_class_mu = zeros(d, C);
each_class_cov = zeros(d, d, C);
mu = mean(X, 2);
S_w = zeros(d, d);

% initialize sample size for each class size
for ii = 1:C
    flag = (Y == ii);
    prior(ii) = sum(flag)/N;
    each_class_size(ii) = sum(flag);    
end

% calculate mean and covariance for each class
for ii = 1:C
   flag = (Y == ii);
   X_c = X(:, flag);
   mu_c = mean(X_c, 2);
   
   each_class_mu(:, ii) = mu_c;
   
   tmp_sum = zeros(d, d);
   for jj = 1:C
      X_c_jj = X_c(:, jj);
      tmp_sum = tmp_sum +  X_c_jj*X_c_jj';
   end
   each_class_cov(:, :, ii) = 1/each_class_size(ii)*tmp_sum - mu_c*mu_c';
end

% calculate within class covariance
for ii = 1:C
   S_w = prior(ii)*each_class_cov(:, :, ii);
end


% calculae between class covariance (total covariance - with class
% covariance)



end