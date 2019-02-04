function data = parseData(filename)
%% function data = parseData(filename)
% function to read the txt file and convert all same level 
% in same number for each column

table = readtable(filename, 'Format','%s%s%s%s%s');
datamat = table2array(table);
[~, ncol] = size(datamat); 

for j = 1:ncol
    
    % unique value preserving order
    uniq = unique(datamat(:,j), 'stable');
    numbericValue = 1:length(uniq);
    
    for i = 1:length(numbericValue)
        datamat = strrep(datamat, uniq(i), num2str(numbericValue(i)));
    end
end

data = str2double(datamat);


    

