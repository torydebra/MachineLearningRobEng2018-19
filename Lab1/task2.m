clearvars
clc
close all

%% TASK 2 plots data and linear regression results

turkishdata= csvread('turkish-se-SP500vsMSCI.csv');
mtcarsdata = csvread('mtcarsdata-4features.csv',1,1);

%% 1) One-dimensional problem without intercept on the Turkish data

x_turkish = turkishdata(:,1);
t_turtkish = turkishdata(:,2);

% Least squares solution to the linear regression problem:
w = ((sum(x_turkish.*t_turtkish))/(sum(x_turkish.^2)));

%Plot result
scatter(x_turkish,t_turtkish, 'x')
hold on
grid on;
% I need only two point to draw a line
% for view reason, i take the min and max value of the data.
xline = [min(x_turkish), max(x_turkish)];
yline = w*xline;
plot (xline, yline)
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Least square solution with whole dataset');

%% 2) Compare graphically the solutions obtained on different random subsets (10%)

xline = [min(x_turkish), max(x_turkish)];
subsetPercentage = 10;
nValue = floor( (length(x_turkish) * (subsetPercentage/100)) );
color ={'r','b','g'};
symbol = {'x', 'o', '^'};
figure

for i = 1:3
    allIndices = randperm(length(x_turkish));
    randomSubset = allIndices(1:nValue);
    
    x = turkishdata(randomSubset,1);
    t = turkishdata(randomSubset,2);

    ws = (sum(x.*t) ) / (sum (x.^2));

    scatter(x,t, symbol{i}, color{i})
    hold on
    yline = ws*xline;
    plot (xline, yline, color{i})
end
grid on
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Least square solutions with 3 different subdataset');

%% 3) One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

x_cars = mtcarsdata(:,4);
t_cars = mtcarsdata(:,1);

w1 = ((sum((x_cars - mean(x_cars)) .* (t_cars - mean(t_cars)))) / (sum((x_cars - mean(x_cars)).^2)));
w0 = mean(t_cars) - w1 * mean(x_cars);

figure
scatter(x_cars,t_cars, 'x')
hold on
grid on;
xline = [min(x_cars), max(x_cars)];
yline = w1.*xline + w0;
plot (xline, yline)
xlabel ("Car Weight (lbs/1000)");
ylabel ("Fuel efficency (mpg)");
title('Least square solution with intercept');

%% 4) Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)
% t : mpg(first column), x : [disp, hp, weight] (2,3,4 columns)

t = mtcarsdata(:,1);
x = mtcarsdata(:,2:4);

% Put a new first column on x is necessary to compute w
% because we have also the intercept calculated with w0
x = [ones(length(x(:,1)),1) x];

% pinv is the moore-penrose pseudoinverse
% we find w0 w1 w2 w3 (3+1 parameters)
% in this case the plot is not drawable, we have 4 dimension so I only
% print the found value of w0 w1 w2 w3
wmul = pinv(x) * t


