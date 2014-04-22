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
   imwrite(face, ['../data/output/eigenface_', num2str(ii), '.png'], 'PNG');
end

max_dim = 286;
precisions = zeros(max_dim, 1);
PCs = PCA(trainX, max_dim);
for ii = 1:max_dim
% feature reduction
T = PCs(:, 1:ii);
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
save('../data/output/eigen_accuracy.mat', 'precisions');


%%
%---------------------------------------------------
% 2. Nonnegative Matrix Factorization
%---------------------------------------------------
close all;
clear all;
% load data 
[trainX trainY] = ReadData('../data/train.txt');
[testX testY] = ReadData('../data/test.txt');
allData = [trainX; testX];
num_nmf_basis = 50;
% first trial 
W = NMF(allData, num_nmf_basis);
for ii = 1:num_nmf_basis
   face = DisplayFace(W(:, ii));
   imwrite(face, ['../data/output/nmfface_round1_', num2str(ii), '.png'], 'PNG');
end

% second trial
close all;
W = NMF(trainX, num_nmf_basis);
for ii = 1:num_nmf_basis
   face = DisplayFace(W(:, ii));
   imwrite(face, ['../data/output/nmfface_round2_', num2str(ii), '.png'], 'PNG');
end

% third trial
close all;
W = NMF(trainX, num_nmf_basis);
for ii = 1:num_nmf_basis
   face = DisplayFace(W(:, ii));
   imwrite(face, ['../data/output/nmfface_round3_', num2str(ii), '.png'], 'PNG');
end

%%
%---------------------------------------------------
% 3. Linear Discriminant Analysis
%---------------------------------------------------
close all;
clear all;

% load data 
[trainX trainY] = ReadData('../data/train.txt');
[testX testY] = ReadData('../data/test.txt');

% PCA feature reduction to 500, in case of singular issue
num_PCAs = 500;
PCs = PCA(trainX, num_PCAs);
trainX_PCA = trainX * PCs;
testX_PCA = testX *PCs;

% Display 10 Fisherfaces
num_fisher_faces = 10;
LDAs = LDA(trainX_PCA, trainY, num_fisher_faces);
LDAs = PCs*LDAs;
for ii = 1:num_fisher_faces
   face = DisplayFace(LDAs(:, ii)); 
   imwrite(face, ['../data/output/fisherface_', num2str(ii), '.png'], 'PNG');
end

max_dim = 286;
precisions = zeros(max_dim, 1);
LDAs = LDA(trainX_PCA, trainY, max_dim);
for ii = 1:max_dim
% feature reduction
T = LDAs(:, 1:ii);
trainX_LDA = trainX_PCA * T;% transpose both side;
testX_LDA = testX_PCA * T;% transpose both side;

% KNN training
knn_model = ClassificationKNN.fit(trainX_LDA, trainY);
y_knn = predict(knn_model, testX_LDA);    

% calculate precisions 
accuracy = sum(testY == y_knn)/length(testY);
fprintf('The accuracy for dim: %d is: %f\n', ii, accuracy);
precisions(ii) = accuracy;
end
save('../data/output/fish_accuracy.mat', 'precisions');



%%
%---------------------------------------------------
% 4. Gaussian Mixture Model
%---------------------------------------------------
close all;
clear all;

% load data 
[trainX trainY] = ReadData('../data/train.txt');
[testX testY] = ReadData('../data/test.txt');
allData = [trainX; testX];

% PCA feature reduction to 500, in case of singular issue
num_PCAs = 50;
PCs = PCA(allData, num_PCAs);
X_PCA = allData * PCs;

num_gaussians = 8;
[w mu sigma] = GMM(X_PCA, num_gaussians);
mu = transpose(mu);

mean_faces = PCs*mu;

for ii = 1:num_gaussians
   face = DisplayFace(mean_faces(:, ii)); 
   imwrite(face, ['../data/output/GMMFace_', num2str(ii), '.png'], 'PNG');
end


