%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [weights mus sigmas] = GMM(X, num_comps)
% Calculate the Gaussian Mixutre Model from unlabeled data
% input:
% X         :   data matrix to be modeled
%
% ouput:
% weights   :   1 by k -- weights for each gaussian components 
% mus       :   k by d -- mean for each component
% sigmas    :   k by d by d -- covariance matrix for each
%               components. 

[N dim] = size(X);
K = num_comps;
weights = ones(1, K)./K;
mus = zeros(K, dim);
sigmas = zeros(K, dim, dim);
Z = zeros(N, K);

stop = 10^(-10);
diff = 10;

% use k-means do initalization for mean
[idx mus] = kmeans(X, K);

for k = 1:K
   flag = (idx == k);
   data = X(flag, :);
   covariance = cov(data);
   sigmas(k, :, :) = covariance;
end

old_likelihood = -10;
new_likelihood = -100;
max_iteration = 1000;

for ii = 1:max_iteration
   % E-step
   for k = 1:K
       w_tmp = weights(k);
       mu_tmp = mus(k, :);
       sigma_tmp = squeeze(sigmas(k, :, :));
       pro = mvnpdf(X, mu_tmp, sigma_tmp);
       Z(:, k) = w_tmp*pro;
   end
   marginal = sum(Z, 2);
   rep_marginal = repmat(marginal, 1, K);
   Z = Z./rep_marginal;
   new_likelihood = sum(log(marginal));
   
   % M-step
   marginal_z = sum(Z, 1);
   % update weights 
   weights = 1/N*marginal_z;
   
   for k = 1:K
       % update mean 
       mu_tmp = zeros(1, dim);
       for jj = 1:N
          mu_tmp = mu_tmp + X(jj, :)*Z(jj, k); 
       end
       mu_tmp = mu_tmp / marginal_z(k);
       mus(k, :) = mu_tmp;
       
       % update sigma
       covariance = zeros(dim, dim);
       for jj = 1:N
           dis = X(jj, :) - mu_tmp;
           covariance = covariance + Z(jj, k)*(dis'*dis);
       end
       sigmas(k, :, :) = covariance/marginal_z(k);
   end
   
   diff = new_likelihood - old_likelihood;
   diff = norm(diff);
   old_likelihood = new_likelihood;
   
   if diff < stop
       break;
   end
end


end