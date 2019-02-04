%% Feature extraction using alexnet
close all;
clearvars;
clc;

% load the sample images as an image datastore labelling automatically 
% them based on folder names 
imds = imageDatastore('101_ObjectCategories', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split the data into 70% training and 30% test data.
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

% numTrainImages = numel(imdsTrain.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(imdsTrain,idx(i));
%     imshow(I)
% end

net = alexnet;
inputSize = net.Layers(1).InputSize; % the image input layer requires input images of size 227-by-227-by-3

% To automatically resize the training and test images before they are input 
% to the network, create augmented image datastores, specify the desired image 
% size, and use these datastores as input arguments to activations
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest, 'ColorPreprocessing', 'gray2rgb');
layer = 'fc8';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

% Extract the class labels from the training and test data
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

% Use the features extracted from the training images as predictor variables
classifier = fitcecoc(featuresTrain,YTrain);

% Classify the test images
YPred = predict(classifier,featuresTest);

% Display 16 sample test images with their predicted labels.
numTestImages = numel(imdsTest.Labels);
idx = randperm(numTestImages,16);
figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(string(label));
end

% Calculate the classification accuracy on the test set.
accuracy = mean(YPred == YTest)