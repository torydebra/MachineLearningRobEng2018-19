clc
clearvars
close all

wineData = importfile('wine.data',14,',');
wineInput = wineData(:,2:end);
wineTargets = oneVsAllProblem(wineData(:,1));

wifiData = importfile('wifi-localization.data',8,'\t');
wifiInput = wifiData(:,1:end-1);
wifiTargets = oneVsAllProblem(wifiData(:,end));

hiddenNeurons = 30;

neuralPatternRecognition(wineInput, wineTargets, hiddenNeurons);
neuralPatternRecognition(wifiInput, wifiTargets, hiddenNeurons);