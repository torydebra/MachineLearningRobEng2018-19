clearvars
clc
close all
addpath('./mnist')


%% Prepare data
[train,labelTrain] = loadMNIST(0);
[test, labelTest] = loadMNIST(1);

%adding last column as target
training = [train, labelTrain];
testing = [test, labelTest];

NTrainTot = length(training(:,1));
NTestTot = length(testing(:,1));

% Choose only fractions of the data (condensed kNN)
nTrain = 1000; %number of test from testing (min is the max k, max is 60000)
nTest = 100; %number of test from testing (max is 10000)
nClass = 10; %number of classes

% Take random row for train and test
trainingSub = training( randi(NTrainTot, [1,nTrain]) ,:) ;
testingSub = testing( randi(NTestTot, [1,nTest]) ,:) ;


%% Classifier with different k value
% I choose k values that are not multipliers of the number of classes to
% avoid ties
kNeighbours = [1:9, 11:5:16, 21:10:40];
nKNeighbours = numel(kNeighbours);
out = zeros(nTest,nKNeighbours);
err = zeros(1,nKNeighbours);
for i = 1:nKNeighbours
	k = kNeighbours(i);
	[out(:,i), err(:,i)] = kNNClassifier(trainingSub, testingSub, k);
end

% Plot error rates
figure
h = bar(err);
h(1).FaceColor = 'g';
set(gca,'xticklabel',kNeighbours)
title('Error rate (number of errors/number of test set rows)')
ylabel('Error')
xlabel('Number of Neighbours')


%% One-VS-all problem
% each column of vsAll matrix will refer to a class, each row to a observation of
% the test set. If vsAll(i,j) == 1, then observation i is classified as class j,
% otherwise it is 0 and the observation isn't classified as class j,
vsAll = zeros(nTest, nClass,nKNeighbours);
label = [1:nClass];

for i = 1:nKNeighbours
    for obs = 1:nTest
        vsAll(obs,:,i) = (out(obs,i)==label);
    end

end
figure;
% Plot classifications for a value of k = 4
h = heatmap(vsAll(:,:,4));
title(['1VsAll problems for each class, with 4 Neighbour', newline,  ...
    'Full blue box means that the test has been classified as that class']);
xlabel('Classes')
ylabel('Tests')
h.ColorbarVisible = 'off';


%% Accuracy, Sensitivity, Sensibiliy and Precision
truePos = zeros(nKNeighbours,nClass);
trueNeg = zeros(nKNeighbours,nClass);
falsePos = zeros(nKNeighbours,nClass);
falseNeg = zeros(nKNeighbours,nClass);

for k = 1:nKNeighbours
    for class = 1:nClass
        for obs = 1:nTest

            if (testingSub(obs,end) == class)
                if (vsAll(obs,class,k) == 1)
                    truePos(k,class) = truePos(k,class) + 1;
                else
                    falseNeg(k,class) = truePos(k,class) + 1;
                end
            else
                if (vsAll(obs,class,k) == 1)
                    falsePos(k,class) = truePos(k,class) + 1;
                else
                    trueNeg(k,class) = truePos(k,class) + 1;
                end   
            end
            
        end
    end
end

Accuracy = (truePos + trueNeg) ./ (truePos + trueNeg + falsePos + falseNeg);
Sensitivity = truePos ./ (truePos + falseNeg);
Specificity = trueNeg ./ (trueNeg + falsePos);
Precision = truePos ./ (truePos + falsePos);

% Plots 
figure
l{1} = 'Accuracy';
l{2} = 'Sensitivity';
l{3} = 'Specificity';
l{4} = 'Precision';
for p =1:6
    subplot(3,2,p)
    h = bar([Accuracy(:,p).*100, Sensitivity(:,p).*100, Specificity(:,p).*100,Precision(:,p).*100]);
    ylim([0 105])
    ytickformat('%g %%')
    xlabel('Number of Neighbours')
    h(1).FaceColor = 'r';
    h(2).FaceColor = 'g';
    h(3).FaceColor = 'b';
    h(4).FaceColor = 'y';    
    set(gca,'xticklabel',kNeighbours)
    title(['Results for class ', num2str(p)]);
end
legend(h,l);

figure
for p =7:10
    subplot(3,2,(p-6))
    h = bar([Accuracy(:,p).*100, Sensitivity(:,p).*100, Specificity(:,p).*100,Precision(:,p).*100]);
    ylim([0 105])
    ytickformat('%g %%')
    xlabel('Number of Neighbours')
    h(1).FaceColor = 'r';
    h(2).FaceColor = 'g';
    h(3).FaceColor = 'b';
    h(4).FaceColor = 'y';    
    set(gca,'xticklabel',kNeighbours)
    title(['Results for class ', num2str(p)]);
end
legend(h,l);
