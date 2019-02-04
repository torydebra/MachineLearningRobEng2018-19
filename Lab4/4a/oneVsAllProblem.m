function vsAll = oneVsAllProblem(targets)
%% function vsAll = oneVsAllProblem(targets)
% This function returns a matrix containing the one-vs-all problem (if
% there is a class = 0 we substitute 0 with nClass so we can indices the
% matrix
% Input:
%   targets = column of the data set containing classes
%
% Output:
%   vsAll = each column of vsAll matrix will refer to a class, each row to 
%           a observation of the test set. If vsAll(i,j) == 1, then 
%           observation i is classified as class j, otherwise it is 0 and 
%           the observation isn't classified as class j.

nIstances = length(targets);
nClass = length(unique(targets));
vsAll = zeros(nIstances, nClass);
label = 1:nClass;
class = find(targets == 0);
if (length(class)) ~= 0
    targets(class) = nClass;
end

for ist = 1:nIstances
    vsAll(ist,:) = (targets(ist)==label);
end