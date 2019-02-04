clearvars
clc
close all
addpath('./mnist')


%% Prepare data
% extract training data from MNIST, second argument is to choose which
% class extract (10 for handwritten 0)
[trainA,labelTrainA] = loadMNIST(0,9);
[trainB,labelTrainB] = loadMNIST(0,5);
NTrainA = length(trainA(:,1));
NTrainB = length(trainB(:,1));

% put togheter observations and label so with rand I extract random row
% with correspondent label
trainA = [trainA, labelTrainA];
trainB = [trainB, labelTrainB];

nTrain = 100;
trainAsub = trainA( randi(NTrainA, [1,nTrain]) ,:) ;
trainBsub = trainB( randi(NTrainB, [1,nTrain]) ,:) ;
training = [trainAsub(:,1:end-1); trainBsub(:,1:end-1)];

labelTrain = [trainAsub(:,end); trainBsub(:,end)];

%% Autoencoder
nh = 2 ;%number of hidden units, 2 so I can plot learning into a 2D plot
myAutoencoder = trainAutoencoder(training',nh);
myEncodedData = encode(myAutoencoder,training');

%% Plot results
plotcl(myEncodedData', labelTrain)
xlabel('Hidden unit 1');
ylabel('Hidden unit 2');
title(['Output of the autoencoder', newline, ...
    'with ', num2str(nTrain), ' instances of classes ', num2str(labelTrainA(1)), ' and ', num2str(labelTrainB(1))]);
legend(['Class ', num2str(labelTrainA(1))], ['Class ', num2str(labelTrainB(1))]);
