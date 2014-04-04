%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

function [W] = NMF(X, num_basis)
% Calculate the NMF bases
% input:
% X         :   data matrix to be transformed 
% num_bases :   number of basis needed
%
% ouput:
% W         :   basis matrix in size: d by num_basis

X = double(X);
X = transpose(X);

[d N] = size(X);



% initialization
k = rand()*10;

min_v = k*10;
max_v = k*20;
W = min_v + (max_v - min_v)*rand(d, num_basis, 'double');
H = min_v + (max_v - min_v)*rand(num_basis, N, 'double');

max_iter = 10000;
threshold = 1;


for ii = 1:max_iter
   H_tmp = H;
   W_tmp = W;

   m1 = (W'*X)./(W'*W*H);
   H = H.*m1;
   
   m2 = (X*H')./(W*H*H');
   W = W.*m2;
   
   diff_H = norm(H - H_tmp);
   diff_W = norm(W - W_tmp);
    
   fprintf('The %dth iteration: diff_W is %f diff_H is %f \n', ii, diff_W, diff_H);
   
   if diff_H <= threshold && diff_W <= threshold
       return 
   end
   
end

end