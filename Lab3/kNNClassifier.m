function [out, err] = kNNClassifier(train, test, k)
%% function [out, err] = kNNClassifier(train, test, k)
% This function classifies each row of the test set according to kNN
% Classifier rule based on the train tesk. It also return the error if the
% true labels for the test are provided 
% Input :
%   train: (n x d) matrix of training data
%   test: (m x c) matrix of testing data, with c >= d-1
%   
% Output:
%   out: (m x 1) vector which classify the test data
%   err: 0(min err) to 1(max err) scalar find as 
%         (number of classifying errors / m)
%         if c == d-1 err can not be compute and will be NaN 
%%

[nRowTrain, nColTrain] = size(train);
[nRowTest, nColTest] = size(test);
nClasses = max(unique(train(:,end)));

%% Checking inputs
if (nColTest < nColTrain -1)
    error("Number of columns of test data must be at least number of column of the training data - 1");
end

if (k<=0 || k > nRowTrain)
    error("Not a valid K");
end

if ( mod(k, nClasses) == 0 )
    disp("K should not be divisible by the number of the classes");
    out = NaN;
    err = NaN;
    return
end

out = zeros(nRowTest,1);

%% Classifier
for query = 1:nRowTest
    
    % I use the euclidian norm to calculate the distance, but others
    % distances could be used
    % vecnorms calculate norms on the rows
    norms = vecnorm(train(:, 1:end-1) - test(query, 1:(nColTrain-1) ), 2, 2);
    n = zeros(k,1);
    
    % Computationally is more efficent do the min() k times if
    % k<log2(nRowTrain) because using min complexity is O(nk), 
    % instead complexity of sorting is O(nLog2(nRowTrain)) 
    if k<log2(nRowTrain)
        for ki = 1:k
            [~,n(ki)] = min(norms);
            norms(n(ki)) = Inf; % so next iteration this value isn't taken into account
        end
    else
        % Ordering to find the neighbours
        % I am interested in the indexes and not in the ordered values of norms
        [~, sortedIndex] = sort(norms);
         % Choose the first k neighbours
         n = sortedIndex(1:k);
     end
  
    % Take the most frequently value of the first k neighbours
    out(query) = mode(train(n,end), 1);
    
end

%% Err
% I compute the error only if columns of test are >= than the column 
% of the train
% if the train has less col, this function throws an error at the beginning
% if it has more, I assume that the real target is in the nColTest column 
% of test data
err = NaN; %return nan for error if nColTest ~= nColTrain

if nColTest >= nColTrain % I can compute the error
    % max error is 1 (every prediction is wrong)
    % NaN value in out is considered an error
    err = (sum(out ~= test(:,nColTrain))) / nRowTest;
end
