%% Train Deep Learning Network to Classify New Images
close all;
clearvars;
clc;

% load the sample images as an image datastore labelling automatically 
% them based on folder names
imds = imageDatastore('101_ObjectCategories2', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% Split the data into 70% training and 30% test data.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

net = googlenet;
% analyzeNetwork(net);
inputSize = net.Layers(1).InputSize; % This layer requires input images of size 224-by-224-by-3

% Extract the layer graph from the trained network. If the network is a 
% SeriesNetwork object, such as AlexNet, VGG-16, or VGG-19, then convert 
% the list of layers in net
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Find the names of the two layers to replace
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

% The last layer with learnable weights is a fully connected layer. Replace 
% this fully connected layer with a new fully connected layer with the number 
% of outputs equal to the number of classes in the new data set (5, in this example)
numClasses = numel(categories(imdsTrain.Labels));
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
plot(lgraph)
title('Last layers of the original GoogLeNet');
ylim([0,10])
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% The classification layer specifies the output classes of the network. 
% Replace the classification layer with a new one without class labels. 
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer 
% graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
title('Last layers of the replaced network');
ylim([0,10])
layers = lgraph.Layers;
connections = lgraph.Connections;

% Extract the layers and connections of the layer graph and select which 
% layers to freeze. In GoogLeNet, the first 10 layers make out the initial 
% 'stem' of the network. Use the supporting function freezeWeights to set 
% the learning rates to zero in the first 10 layers. Use the supporting 
% function createLgraphUsingConnections to reconnect all the layers in the 
% original order. The new layer graph contains the same layers, but with 
% the learning rates of the earlier layers set to zero.
layers(1:140) = freezeWeights(layers(1:140));
lgraph = createLgraphUsingConnections(layers,connections);

% To automatically resize the training and test images before they are input 
% to the network, create augmented image datastores, specify the desired image 
% size.
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter, 'ColorPreprocessing', 'gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation, 'ColorPreprocessing', 'gray2rgb');

% Train the network using the training data. By default, trainNetwork uses 
% a GPU if one is available, otherwise trainNetwork uses a CPU.
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options);

% Classify the validation images using the fine-tuned network, and calculate 
% the classification accuracy.
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

% Display four sample validation images with predicted labels and the 
% predicted probabilities of the images having those labels.
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end