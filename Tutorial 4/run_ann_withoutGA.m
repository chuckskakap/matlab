function results = run_ann_withoutGA()

% set RNG seed number to get reproducible results; 
% change seed number to get different results
rng('default');
rng(4);

% load dataset
data = dlmread('iris.csv');

% dataset has four attributes and three classes
X = data(:, 1:4);
Y = data(:, 5);

% initialize network; make sure the network's first layer has the same
% number of nodes as the number of attributes, and the last layer has the
% same number of nodes as the number of classes

net = create_network();

% test the untrained network and get accuracy
Y_pred = test(net, X);
results = mean(Y==Y_pred');
disp("Accuracy is " + string(results));

% accuracy may be very low; how to improve the results?

end