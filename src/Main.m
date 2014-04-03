%---------------------------------------------------
% author    : SHI XUDONG
% title     : Age Face Recognition main code
% date      : 2014.04.01
%---------------------------------------------------

cd ../data;
addpath(pwd);
cd ../src;
close all;
clear all;

%%
%---------------------------------------------------
% 1. Principle Component Analysis
%---------------------------------------------------
close all;
clear all;
% load data 
[trainX trainY] = ReadData('../data/train.txt');
[testX testY] = ReadData('../data/test.txt');

num_eig_faces = 10;
PCs = PCA(trainX, num_eig_faces);
for ii = 1:num_eig_faces
   face = DisplayFace(PCs(:, ii)); 
   imwrite(face, ['../data/ouput/eigenface_', num2str(ii), '.png'], 'PNG');
end

max_dim = 286;
precisions = zeros(max_dim, 1);
for ii = 1:max_dim
% feature reduction
T = PCA(trainX, ii);
trainX_PCA = trainX * T;% transpose both side;
testX_PCA = testX * T;% transpose both side;

% KNN training
knn_model = ClassificationKNN.fit(trainX_PCA, trainY);
y_knn = predict(knn_model, testX_PCA);    

% calculate precisions 
accuracy = sum(testY == y_knn)/length(testY);
fprintf('The accuracy for dim: %d is: %f\n', ii, accuracy);
precisions(ii) = accuracy;
end
save('../data/ouput/eigen_accuracy.mat', 'precisions');


%%
%---------------------------------------------------
% 2. Nonnegative Matrix Factorization
%---------------------------------------------------
close all;
clear all;
% load data 
[trainX trainY] = ReadData('../data/train.txt');
num_nmf_basis = 50;
W = NMF(trainX, num_nmf_basis);
for ii = 1:num_nmf_basis
   face = DisplaySequences(W(:, ii));
   imwrite(face, ['../data/output/nmf_face_', num2str(ii), '.png'], 'PNG');
end


%%