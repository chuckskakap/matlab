function [population, fitness_score, progress] = run_ann_withGA_model1()

rng('default'); %seed also need to change at create_network
rng(17); % must same seed

% load dataset
data = readmatrix('clean_seed_dataset.csv');

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

input_layer_units = 7; % how many input features?

max_hidden_layer = 6;

output_layer_units = 3; % how many outputs?


hidden_layers_fitness = [0, 0, 0, 0, 0, 0]; 

% set parameters for GA
population_size = 50; % 50 chromosomes for population
generations_max = 250; % run for 50 generations

progress = [];

current_layer = 1;
best_num_hidden_layer = 1;
while current_layer <= max_hidden_layer
    disp("****** NUMBER OF HIDDEN LAYER: " + current_layer + " ******");
    
    hidden_layer_units = GA_hidden_layers_units(population_size, generations_max, current_layer, X, Y); % hlu = hidden layer units

    % create network based on the best hidden layer units found
    net = create_network(input_layer_units, current_layer, hidden_layer_units, output_layer_units);

    %GA to optimize weight and biases based on the network just created
    [fitness_score,population] = GA_weight_bias(net, population_size, generations_max, current_layer, X, Y);
    
    if fitness_score > max(hidden_layers_fitness)
        best_num_hidden_layer = current_layer;
    end
    hidden_layers_fitness(current_layer) = fitness_score;

    disp("THE BEST NUMBER OF HIDDEN LAYER " +best_num_hidden_layer);
    disp("THE MAX SCORE " +max(hidden_layers_fitness));
    
    current_layer = current_layer + 1;
end

end



function best_hidden_layer_units = GA_hidden_layers_units(population_size, generations_max, current_num_of_layers, X, Y) %hlu = hidden layer units
    disp("****** OPTIMIZATION FOR " + current_num_of_layers + " LAYERS UNITS")

    convergence_maxcount = 10; % stop the GA if the average fitness score stopped increasing for 5 generations
    convergence_count = 0;
    convergence_avg = 0;
    selrate = 0.5; % SelectionRate
    mutrate = 0.5; % MutationRate
    input_layer_units = 7; 
    output_layer_units = 3; 

    progress = [];
    population_fitness_score_container = containers.Map();  

    population = randi([1,20], population_size, current_num_of_layers);
    fitness_score = zeros(population_size, 1);
    fit_max = max(fitness_score);

    generations_current = 1;
    while generations_current <= generations_max
        for i = 1 : population_size
            
            hidden_layer_units  = fix(population(i, :));
            net = create_network(input_layer_units, current_num_of_layers, hidden_layer_units, output_layer_units);
            

            if fitness_score(i,1) == 0
                fitness_score(i,1) = fitness_function_hidden_layer_unit(net, X, Y);

                if fitness_score(i,1) >= fit_max
                    fit_max = fitness_score(i,1);
                    best_hidden_layer_units = hidden_layer_units;
                end
            end

            floatStrKey = num2str(fitness_score(i,1));
            arrayStrKey = mat2str(hidden_layer_units);
            population_fitness_score_container(floatStrKey) = arrayStrKey;
        end
    
        fit_avg = mean(fitness_score);
        fit_max = max(fitness_score);
        progress = [progress; fit_avg; fit_max];
    
        % convergence? 
        if fit_avg > convergence_avg
            convergence_avg = fit_avg;
            convergence_count = 0;
            disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
        else
            convergence_count = convergence_count + 1;
        end
    
        generations_current = generations_current + 1;
        if (fit_max >= 1)
            generations_max = 0;
            disp("Reached convergence.")
            disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
        
        elseif (convergence_count > convergence_maxcount)
            % what to do if fitness haven't improved?
            % stop the GA?
            % generations_max = 0;
            % disp("Reached convergence.")
    
            % or adjust selection rate and mutation rate for fine-grained search
            if (selrate < 0.9)
                convergence_count = 0;
                selrate = selrate + 0.1;
                mutrate = mutrate - 0.1;
            else
                generations_max = 0;
                disp("Reached convergence.")
                disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));

            end
        end
        [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate);    
        

    end

    str_fit_max = num2str(fit_max);
    if isKey(population_fitness_score_container, str_fit_max)
        value = population_fitness_score_container(str_fit_max);
        fprintf('THE BEST FITNESS SCORE FOR : %s, HIDDEN LAYER UNITS : %s\n', str_fit_max, value);
    end

end





function [fit_max, population] = GA_weight_bias(net, population_size, generations_max, current_num_of_layers, X, Y)
    
    disp("****** OPTIMIZATION WEIGHT AND BIASES FOR " + current_num_of_layers + " HIDDEN LAYERS ******")
    convergence_maxcount = 25; 
    convergence_count = 0;
    convergence_avg = 0;
    selrate = 0.5; % SelectionRate
    mutrate = 0.5; % MutationRate
    progress = [];

    % initialize population
    num_genes = sum(net.num_parameters); 
    population = rand(population_size, num_genes) * 2 - 1;
    fitness_score = zeros(population_size, 1);
    
    generations_current = 1;
    while generations_current < generations_max
        for i = 1:population_size
            if fitness_score(i,1) == 0
                fitness_score(i,1) = fitness_function_weight_bias(net, population(i, 1:end), X, Y);
            end
        end
    
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
        if (fit_max >= 1)
            generations_max = 0;
            disp("Reached convergence.")
            disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
        
        elseif (convergence_count > convergence_maxcount)
            % what to do if fitness haven't improved?
            % stop the GA?
            % generations_max = 0;
            % disp("Reached convergence.")
            % disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
            % 
            % or adjust selection rate and mutation rate for fine-grained search
            if (selrate < 0.9)
                convergence_count = 0;
                selrate = selrate + 0.1;
                mutrate = mutrate - 0.1;
            else
                generations_max = 0;
                disp("Reached convergence.")
                disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
            end
        end
    
        % do genetic operators
        [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate);
        
    end
end



function score = fitness_function_hidden_layer_unit(net, X, Y)

% test network
Y_pred = test(net, X);

% fitness score is the accuracy of the prediction
score = mean(Y == Y_pred');

end

function score = fitness_function_weight_bias(net, chromosome, X, Y)

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