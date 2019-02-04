%% Clasify images with googlenet
close all;
clearvars;
clc;

net = googlenet;

inputSize = net.Layers(1).InputSize;
classNames = net.Layers(end).ClassNames;

myFolder = strcat(pwd,'/coil-100');
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
    uiwait(warndlg(errorMessage));
    return;
end
filePattern = fullfile(myFolder, '*.png');
jpegFiles = dir(filePattern);

% classify all the images of the folder 
% (step=72 because are 72 images of the same category consecutively)
for k = 1:72:length(jpegFiles)
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    imageArray = imread(fullFileName);
    
    %drawnow; % Force display to update immediately.
    
    imageArray = imresize(imageArray,inputSize(1:2));
    imshow(imageArray);  % Display image.

    [label,scores] = classify(net,imageArray);

    % plot the image to classify
    subplot(2,1,1)
    imshow(imageArray)
    title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");

    [~,idx] = sort(scores,'descend');
    idx = idx(5:-1:1);
    classNamesTop = net.Layers(end).ClassNames(idx);
    scoresTop = scores(idx);

    % plot the top 5 predictions
    subplot(2,1,2)
    barh(scoresTop)
    xlim([0 1])
    title('Top 5 Predictions')
    xlabel('Probability')
    yticklabels(classNamesTop)
    
    disp('Press any key to classify another image');
    pause;
end