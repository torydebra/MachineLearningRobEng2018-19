clearvars
clc
close all

%% TASK 3 compare results using mean square error
turkish = csvread('turkish-se-SP500vsMSCI.csv');
mtcarsdata = csvread('mtcarsdata-4features.csv',1,1);

%% Re-runned 1,3 and 4 from task 2 using only 50% of the data

subsPerc = 50/100;
nValueTur = floor(length(turkish(:,1))*subsPerc);
nValueCar = floor(length(mtcarsdata(:,1))*subsPerc);

for i=1:10
    allIndicesTur = randperm(length(turkish(:,1)));
    randomSubsetTur = allIndicesTur(1:nValueTur);

    x_turSub = turkish(randomSubsetTur,1);
    t_tur = turkish(randomSubsetTur,2);
    % Least squares solution to the linear regression 
    wTur = ((sum(x_turSub.*t_tur))/(sum(x_turSub.^2)));
    
    % Compute the objective (mean square error) on the subset
    y_turSub = wTur * x_turSub;
    J_MSE_tur = mean((t_tur - y_turSub).^2);
    
    % Compute the objective of the same models on the other subset
    randomSubsetTur2 = allIndicesTur(nValueTur+1:end);

    x_turSub2 = turkish(randomSubsetTur2,1);
    t_turSub2= turkish(randomSubsetTur2,2);
    y_turSub2 = wTur * x_turSub2;
    
    % Mean square error objective
    J_MSE_turSub2 = mean((t_turSub2 - y_turSub2).^2);
    
    % Vector for the bar graph
    J_MSE_tur_bar(i,:) = [J_MSE_tur J_MSE_turSub2];
end

% Plot bar graph for turkish data
figure
h1 = bar(J_MSE_tur_bar);
h1(1).FaceColor = 'b';
h1(2).FaceColor = 'c';
title('Mean square error on turkish data')
ylabel('J-MSE')
xlabel('10 different training-test random splits')
l1{1} = 'First 50% of the data'; l1{2} = 'Remaining 50% of the data';
legend(h1,l1);

% 3) One-dimensional problem with intercept on the Motordata, 
% using columns mpg (as target) and weight (as observation)
for j=1:10
    
    allIndicesCar = randperm(length(mtcarsdata(:,1)));
    randomSubsetCar = allIndicesCar(1:nValueCar);
    
    x_carsSub = mtcarsdata(randomSubsetCar,4);
    t_carsSub = mtcarsdata(randomSubsetCar,1);
    
    w1Car1 = (sum( (x_carsSub - mean(x_carsSub)) .* (t_carsSub - mean(t_carsSub)) ) ) / (sum( (x_carsSub - mean(x_carsSub)).^2) ); 
    w0Car1 = mean(t_carsSub) - w1Car1 * mean(x_carsSub);

    % Compute the objective (mean square error) on the training data
    y_carsSub = w1Car1 * x_carsSub + w0Car1;
    J_MSE_carsSub = mean((t_carsSub - y_carsSub).^2);

    % Compute the objective of the same models on the remaining 50% of the data
    randomSubsetCar2 = allIndicesCar(nValueCar+1:end);

    x_carsSub2 = mtcarsdata(randomSubsetCar2,4);
    t_carsSub2 = mtcarsdata(randomSubsetCar2,1);
    y_carssub2 = w1Car1 * x_carsSub2 + w0Car1;
    
    % Mean square error objective
    J_MSE_carsSub2 = mean((t_carsSub2 - y_carssub2).^2);
    
    % Vector for the bar graph
    J_MSE_cars_bar(j,:) = [J_MSE_carsSub J_MSE_carsSub2];
end

% Bar graph for cars data
figure
h2 = bar(J_MSE_cars_bar);
h2(1).FaceColor = 'r';
h2(2).FaceColor = 'y';
title('Mean square error on one-dimensional problem of cars data')
ylabel('J-MSE')
xlabel('10 different training-test random splits')
l2{1} = 'First 50% of the data'; l2{2} = 'Remaining 50% of the data';
legend(h2,l2);


% 4) Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)
% t : mpg(first column), x : [disp, hp, weight] (2,3,4 columns)
for k=1:10
    
    allIndicesCar2 = randperm(length(mtcarsdata(:,1)));
    randomSubsetCar2 = allIndicesCar2(1:nValueCar);

    tSub = mtcarsdata(randomSubsetCar2,1);
    xSub = mtcarsdata(randomSubsetCar2,2:4);

    % Put a new first column on x is necessary to compute w
    % because we have also the intercept calculated with w0
    xSub = [ones(length(xSub(:,1)),1) xSub];

    % pinv is the moore-penrose pseudoinverse
    % we find w0 w1 w2 w3 (3+1 parameters)
    wmul = pinv(xSub) * tSub;

    % Compute the objective (mean square error) on the training data
    ySub = xSub * wmul;
    J_MSE_cars_multi = mean((tSub - ySub).^2);

    % Compute the objective of the same models on the remaining 50% of the data
    randomSubsetCarSub2 = allIndicesCar2(nValueCar+1:end);

    tSub2 = mtcarsdata(randomSubsetCarSub2,1);
    xSub2 = mtcarsdata(randomSubsetCarSub2,2:4);

    % Put a new first column on x is necessary to compute w
    % because we have also the intercept calculated with w0
    xSub2 = [ones(length(xSub2(:,1)),1) xSub2];
    
    % Compute the objective (mean square error) on the other subset
    ySub2 = xSub2 * wmul;
    J_MSE_cars_multiSub2 = mean((tSub2 - ySub2).^2);
    
    % Vector for the bar graph
    J_MSE_cars_multi_bar(k,:) = [J_MSE_cars_multi J_MSE_cars_multiSub2];
    
end

% Bar graph for multi-dimensional problem of cars data
figure
h3 = bar(J_MSE_cars_multi_bar);
h3(1).FaceColor = 'g';
h3(2).FaceColor = 'm';
title('Mean square error on multi-dimensional problem of cars data')
ylabel('J-MSE')
xlabel('10 different training-test random splits')
l3{1} = 'First 50% of the data'; l3{2} = 'Remaining 50% of the data';
legend(h3,l3);