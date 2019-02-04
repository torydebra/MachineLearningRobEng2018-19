clearvars
clc
close all

mtcars = csvread("mtcarsdata-4features.csv",1,1);
turkish = csvread("turkish-se-SP500vsMSCI.csv");

%Compute cost function with MSE
%J = immse(turkish(:,1),turkish(:,2)) %Requires image processing toolbox

%% TASK2.1
w = (sum(turkish(:,1).*turkish(:,2)) ) / (sum (turkish(:,1).^2));

scatter(turkish(:,1),turkish(:,2), 'x')
hold on
grid on;
xline = [-0.06, 0.08];
yline = w*xline;
plot (xline, yline)
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Least square solution with whole dataset');


%% TASK2.2
% xline = [-0.06, 0.08];
% subsetPercentage = 10;
% nValue = floor( (length(turkish(:,1)) * (subsetPercentage/100)));
% color ={'r','b','g'};
% symbol = {'x', 'o', '^'};
% figure
% grid on;
% 
% for i = 1:3
%     allIndices = randperm(length(turkish(:,1)));
%     randomSubset = allIndices(1:nValue);
%     
%     x = turkish(randomSubset,1);
%     t = turkish(randomSubset,2);
% 
%     ws = (sum(x.*t) ) / (sum (x.^2));
% 
%     scatter(x,t, symbol{i}, color{i})
%     hold on
%     yline = ws*xline;
%     plot (xline, yline, color{i})
% end
% xlabel ("Standard and Poor's 500 return index");
% ylabel ("MSCI Europe index");
% title('Least square solutions with 3 different subdataset (each one 10% of the whole set)');

xline = [-0.06, 0.08];
subsetPercentage = 10;
nValue = floor( (length(turkish(:,1)) * (subsetPercentage/100)));
color ={'r','b','g'};
symbol = {'x', 'o', '^'};
figure

for i = 1:3
    allIndices = randperm(length(turkish(:,1)));
    randomSubset = allIndices(1:nValue);
    
    x = turkish(randomSubset,1);
    t = turkish(randomSubset,2);

    w1 = (sum ((x - mean(x)) .* (t - mean(t))) ) / (sum ((x -mean(x)).^2) );

    w0 = mean(t) - w1*mean(x);

    scatter(x,t, symbol{i}, color{i})
    hold on
    
    yline = w1.*xline + w0;
    plot (xline, yline, color{i})
end
grid on
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Least square solutions with 3 different subdataset (each one 10% of the whole set)');


%% TASK2.3
% Interception on single dimension
% x : mpg(first column), t : weight (4 columns)
t = mtcars(:,1);
x = mtcars(:,4);

w1 = (sum ((x - mean(x)) .* (t - mean(t))) ) / (sum ((x -mean(x)).^2) );

w0 = mean(t) - w1*mean(x);

figure
scatter(x,t, 'x')
hold on
grid on;
xline = [0, 5.5];
yline = w1.*xline + w0;
plot (xline, yline)
xlabel ("Car Weight (lbs/1000)");
ylabel ("Fuel efficency (mpg)");
title('Least square solution with interception');


%% TASK2.4
% Interception on multidimensional
% t : mpg(first column), x : [disp, hp, weight] (2,3,4 columns)

t = mtcars(:,1);
x = mtcars(:,2:4);

% Put a new first column on x is necessary to compute w
% because we have also the interception calculated with w0

x = [ones(length(x(:,1)),1) x];

% pinv is the moore-penrose pseudoinverse
% we find w0 w1 w2 w3 (3+1 parrameters)
wmul = pinv(x) * t
