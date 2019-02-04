clearvars
clc
close all

datanum = parseData("weather_data_set");
nValue = 10; %number of obs of train set
nTimes = 10; %number of experiments
nTrust = 3; %number of experiment with diffent trustiness (for lapla smoothing)

% number of levels needed for Laplacian smoothing (task 3)
for i = 1:length(datanum(1,:))
    nLevels(i) = length(unique(datanum(:,i)));
end

for i = 1:nTimes
	allIndices = randperm(length(datanum(:,1)));
    randomSubsetTrain = allIndices(1:nValue);
    randomSubsetTest = allIndices(nValue+1:end);
    
	training = datanum(randomSubsetTrain,:);
	testing  = datanum(randomSubsetTest,:);


	%% TASK 2 Naive Bayes classifier

    [out, err] = naiveBayesClass(training, testing, 0);
    fprintf("Classifier without Laplace smoothing:\n");
    fprintf('Classification is %g %g %g %g, With error: %g\n', out, err);


    %% TASK  3 Naive Bayes classifier with laplacian smoothing
    fprintf("Classifier with Laplace smoothing:\n");
    for a = 1:nTrust
        trustiness = 0.5*a; %try with 0.5, 1, 1.5
        % trustiness > 1 means that I trust my prior belief more than the data
        % trustiness < 1 means that I trust my prior belief less than the data
        [outLa, errLa] = naiveBayesClass(training, testing, 1, nLevels, trustiness);
        
        
        fprintf('With a = %g, Classification is %g %g %g %g, With error: %g\n',trustiness, outLa, errLa);
        errorClassLapBar(1,a) = errLa;
    end
    % print real target 
    fprintf('Real target value were: %g %g %g %g\n\n',testing(:,end));

    
    % store errors for plot
    errorClassBar (i,:) = [err errorClassLapBar];
end

figure
h1 = bar(errorClassBar);
h1(1).FaceColor = 'b';
h1(2).FaceColor = 'y';
h1(3).FaceColor = 'g';
h1(4).FaceColor = 'c';
title('Error rate (number of errors/number of test set rows)')
ylabel('Error')
xlabel(sprintf('%d different test set random splits', nTimes))
l1{1} = 'Classifier without Laplace smoothing'; l1{2} = 'Classifier without Laplace smoothing (a=0.5)';
l1{3} = 'Classifier without Laplace smoothing (a=1)'; l1{4} = 'Classifier without Laplace smoothing (a=1.5)';
legend(h1,l1);
