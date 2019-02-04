function [out, err] = naiveBayesClass(train, test, lapla, nLevels, trustiness)
%% function [out, err] = naiveBayesClass(train, test, lapla, nLevels, trustiness)
% This function implements the naive bayes classifier 
% with laplacian smoothing based on train data
% and then classify the test data according to the inferred rule, giving
% also the error of this classification
% Input :
%   train: (n x d) matrix of training data
%   test: (m x c) matrix of testing data, with c >= d-1
%   lapla: 1 for Laplacian smoothing, 0 for vanilla Classifier
%   nLevels(only for lapla =1): (1 x d) vector which says the number of possible values of the
%       attributes
%   trustiness(only for lapla =1): a scalar which represent the trustiness on the data:
%       trustiness > 1 means that I trust my prior belief more than the data
%       trustiness < 1 means that I trust my prior belief less than the data
% Output:
%   out: (m x 1) vector which classify the test data
%   err: 0(min err) to 1(max err) scalar find as 
%         (number of classifying errors / m)
%         if c ~= d err can not be compute without ambiguity and will be NaN

[nRowTrain, nColTrain] = size(train);
[nRowTest, nColTest] = size(test);

% Number of features of the target (for weather data it is 2: 'yes' and 'no')
nFeatTarg = length(unique(train(:,end)));

% Checking inputs
if (nColTest < nColTrain -1)
    error("Number of columns of test data must be at least number of column of the training data - 1");
end
if ( sum(sum(train<1)) ~= 0)
    error("Value in the train matrix must be all greater or equal than 1");
end
if ( sum(sum(test<1)) ~= 0)
    error("Value in the test matrix must be all greater or equal than 1");
end
if lapla == 1
    if nargin < 5
        error("Not enough argument for laplacian smoothing");
    end
end

%% Training of Naive Bayes classifier
% on train data, last column is the target

for ft = 1:nFeatTarg %ft : featTarg
    Nw(ft) = (sum(train(:,end) == ft));
    P(ft) = Nw(ft) / nRowTrain; %prior probability

    for attr = 1:(nColTest -1) % attr: attribute (index for columns)
        for vattr = 1: length(unique(train(:,attr))) % vattr: value of the attribute attr (index for rows)
     
            equivalences = sum(train(:, [attr, end]) == [vattr, ft], 2);
            % if equivalence == 2 there is a match between ft and vattr.
            % sum(equivalences == 2) is the number of observations in ft
            % (the class) where attribute attr has value vattr
            
            if lapla == 1
                % trustiness and nlevel are the correction of the laplace
                % smoother
                W(vattr, attr, ft) = ((sum(equivalences == 2)) + trustiness) /...
                    ( Nw(ft) + trustiness*nLevels(attr));  
            else
                W(vattr, attr, ft) = (sum(equivalences == 2)) / ( Nw(ft) );      
            end
            
        end
    end   
end
    
%% TEST phase

for ft = 1:nFeatTarg %ft : featTarg
    % TEST phase for feature ft of the target
    for row = 1 : nRowTest %row: number of test
        G(row,ft) = P(ft);
        
        % I have to scroll only until column of training test.
        % If the test set has more column, I discard the additional ones
        for i = 1: nColTrain-1
            
           if test(row,i) > length(W(:,1,ft)) || W(test(row,i), i, ft) == 0
               % if so, discard this observation j, it has value that arent in the
               % training set
               G(row,ft) = NaN;
           else
                G(row,ft) = G(row,ft) * W(test(row,i), i, ft) ;
           end
           
        end
    end
end


%% Out
% Compute the maximum among each row of G 
% the position of maximun (first column: prevision =1) is 
% the class. If for some column there is nan value 'omitnan'
% option does not consider it; if all are NaN, that row will be NaN
[~, out] = max(G, [], 2, 'omitnan');


%% Err
% I compute the error only if columns are equal
% if the train has less col, this program throw an error at the beginning
% if it has more, I don't know which one is for the target (the nColTrain or the
% last one?) so i dont provide the error.

err = NaN; %return nan for error if nColTest ~= nColTrain
if nColTest == nColTrain % we can compute the error
    % max error is 1 (every prediction is wrong)
    % NaN value in out is considered an error
    err = (sum(out ~= test(:,end))) / nRowTest;
end
