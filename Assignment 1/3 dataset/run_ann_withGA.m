function [population, fitness_score, progress] = run_ann_withGA()

% set RNG seed number to get reproducible results; 
% change seed number to get different results
rng('default');
rng(1);

% load dataset
data = dlmread('iris.csv');

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

% what to optimize? the neural network weights
% how many values to optimize?
% again, make sure the network's input and output matches the dataset
net = create_network();
num_genes = sum(net.num_parameters);

% set parameters for GA
population_size = 100; % 50 chromosomes for population
generations_max = 1000; % run for 50 generations
selrate = 0.5; % SelectionRate
mutrate = 0.5; % MutationRate
progress = [];

convergence_maxcount = 10; % stop the GA if the average fitness score stopped increasing for 5 generations
convergence_count = 0;
convergence_avg = 0;

% initialize population
population = rand(population_size, num_genes) * 2 - 1;
fitness_score = zeros(population_size, 1);

generations_current = 1;
while generations_current < generations_max
    % test all chromosomes that haven't been tested
    for i = 1:population_size
        if fitness_score(i,1) == 0
            % fitness testing a chromosome
            fitness_score(i,1) = fitness_function(population(i, 1:end), X, Y);
        end
    end

    % find out statistics of the population
    fit_avg = mean(fitness_score);
    fit_max = max(fitness_score);
    progress = [progress; fit_avg, fit_max];

    % convergence? 
    if fit_avg > convergence_avg
        convergence_avg = fit_avg;
        convergence_count = 0;
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    else
        convergence_count = convergence_count + 1;
    end

    generations_current = generations_current + 1;
    % stop the GA if reach 100% accuracy or reach convergence?
    % instead of stopping immediately, slowly adjust SelRate and MutRate
    if (fit_max >= 1)
        generations_max = 0;
        disp("Reached convergence.")
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    
    elseif (convergence_count > convergence_maxcount)
        % what to do if fitness haven't improved?
        % stop the GA?
        generations_max = 0;
        disp("Reached convergence.")
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));

        % % or adjust selection rate and mutation rate for fine-grained search
        % if (selrate < 0.9)
        %     convergence_count = 0;
        %     selrate = selrate + 0.1;
        %     mutrate = mutrate - 0.1;
        % else
        %     generations_max = 0;
        %     disp("Reached convergence.")
        % end
    end

    % do genetic operators
    [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate);
    
end

end



function score = fitness_function(chromosome, X, Y)

net = create_network();
layers = (length(fieldnames(net))-2) / 2;
% set the weights based on the chromosome
for i = 1:layers
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(i);
    
    num_genes = size(net.(layer_name), 1)*size(net.(layer_name), 2);
    new_layer = reshape(chromosome(1:num_genes), [size(net.(layer_name), 1), size(net.(layer_name), 2)]);
    net.(layer_name) = new_layer;
    chromosome = chromosome(num_genes+1:end);

    num_genes = size(net.(bias_name), 1);
    new_bias = chromosome(1:num_genes);
    net.(bias_name) = new_bias';
    chromosome = chromosome(num_genes+1:end);
end

% now test the new network
Y_pred = test(net, X);

% fitness score is the accuracy of the prediction
% what other fitness score can be calculated?
score = mean(Y == Y_pred');

end


function [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate)

% how many chromosomes to reject?
popsize = size(population, 1);
num_reject = round((1-selrate) * popsize);

for i = 1:num_reject
    % find lowest fitness score and remove the chromosome
    [~, lowest] = min(fitness_score);
    population(lowest, :) = [];
    fitness_score(lowest) = [];
end

% for each rejection, create a new chromosome
num_parents = size(population, 1);
for i = 1:num_reject
    % how to select parent chromosomes?
    % random permutation method
    order = randperm(num_parents);
    parent1 = population(order(1), :);
    parent2 = population(order(2), :);

    % mix-and-match
    offspring = (parent1 + parent2) / 2;

    % mutation
    mut_val = rand(1, size(population(1,:), 2));
    mut_val = mut_val * mutrate; 

    for j = 1:size(mut_val, 2)
        if rand < mutrate
            offspring(1, j) = offspring(1, j) + mut_val(1, j);
        end
    end
    
    % add new offspring to population
    population = [population; offspring];
    fitness_score = [fitness_score; 0];
end

end