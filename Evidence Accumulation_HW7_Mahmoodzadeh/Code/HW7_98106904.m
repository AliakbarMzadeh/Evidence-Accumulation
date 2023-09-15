% Q1
% Parameters
num_steps = 1000;  % number of steps in the simulation
dt = 0.01;        % time step
B = 0.5;          % constant bias
sigma = 0.2;      % scaling constant
x0 = 0;           % start point

% Preallocate
X = zeros(1, num_steps);
X(1) = x0;

% Brownian motion
W = zeros(1, num_steps);
W(1) = 0; 

% Simulate evidence accumulation
for t = 2:num_steps
    dW = sqrt(dt) * randn();  % discretized Brownian motion term
    W(t) = W(t-1) + dW;
    dX = B * dt + sigma * dW;  % change in evidence
    X(t) = X(t-1) + dX;
end

% Plot
figure;
plot(0:dt:(num_steps-1)*dt, X);
xlabel('Time');
ylabel('Evidence');
title('Simulation of Evidence Accumulation Model');
grid on; 
grid minor;
%% Q2


% Simulate the 20 choice experiments
B = 1;
sigma = 1;
dt = 0.1;
time_interval = 1;
num_experiments = 20;
choices = zeros(1, num_experiments);
X_values = zeros(num_experiments, round(time_interval / dt));
for i = 1:num_experiments
    [choices(i), X_values(i, :)] = simple_model(B, sigma, dt, time_interval);
end

% Distribution of results
figure;
histogram(choices);
title('Distribution of Choices');
xlabel('Choice');
ylabel('Frequency');

% Time course of the evidence for various B values over a 10-second trial
B_values = [1, 10, 0, 0.1, -1];
time_interval = 10;
num_steps = round(time_interval / dt);
X_values = zeros(length(B_values), num_steps);
for i = 1:length(B_values)
    [~, X_values(i, :)] = simple_model(B_values(i), sigma, dt, time_interval);
end

% Plot the time course of the evidence
figure;
hold on;
for i = 1:length(B_values)
    plot((1:num_steps) * dt, X_values(i, :), 'DisplayName', ['B = ' num2str(B_values(i))]);
end
hold off;
legend;
title('Time Course of the Evidence');
xlabel('Time (s)');
ylabel('Evidence');
grid on;
%% Q3
% Define the parameters
B = 0.1;
sigma = 1;
dt = 0.1;
time_intervals = 0.5:0.5:10;
num_trials = 10;

% Initialize array to hold error rates
error_rates = zeros(1, length(time_intervals));

% Loop over the different time intervals
for i = 1:length(time_intervals)
    num_errors = 0;
    for j = 1:num_trials
        [choice, ~] = simple_model(B, sigma, dt, time_intervals(i));
        if choice <= 0 % A positive response is considered correct
            num_errors = num_errors + 1;
        end
    end
    error_rates(i) = num_errors / num_trials;
end

% Plot the error rates
figure;
plot(time_intervals, error_rates);
title('Error Rate vs Time Interval');
xlabel('Time Interval (s)');
ylabel('Error Rate');

%% Q3
clear; clc; close all;

B = 0.13;          % Bias
sigma = 1;        % Variance
dt = 0.01;         % Time step size
time_intervals = 0.5:0.5:10;  % Time intervals
num_trials = 1000;  % Number of trials

error_rates_sim = zeros(1, length(time_intervals));  % Array to hold simulated error rates
error_rates_th = zeros(1, length(time_intervals));   % Array to hold theoretical error rates

for i = 1:length(time_intervals)
    num_errors = 0;
    for j = 1:num_trials
        T = time_intervals(i);
        [choice, ~] = simple_model(B, sigma, dt, T);
        if choice <= 0  % A positive response is considered correct
            num_errors = num_errors + 1;
        end
    end
    error_rates_sim(i) = num_errors / num_trials;
end

pth_fun = @(t, sigma, T, B) (1/sqrt(2*pi*sigma*T))*exp(-((t - B*T).^2)/(2*sigma*T));

for i = 1:length(time_intervals)
    error_rates_th(i) = integral(@(t) pth_fun(t, sigma, time_intervals(i), B), -inf, 0);
end

% Plot the error rates
figure;
plot(time_intervals, error_rates_sim, '.', 'MarkerSize', 15, 'Color', 'k');
hold on;
plot(time_intervals, error_rates_th, 'k', 'LineWidth', 1.5);
title('Error Rate vs Time Interval');
xlabel('Time Interval (s)');
ylabel('Error Rate');
legend('Simulation', 'Theory');


%% Q4
% Define the parameters
B = 0.1;
sigma = 1;
dt = 0.1;
time_interval = 10;
num_trials = 10;

% Initialize arrays to hold decision variable trajectories, means, and variances
X_values = zeros(num_trials, round(time_interval / dt));
mean_values = zeros(1, round(time_interval / dt));
var_values = zeros(1, round(time_interval / dt));

% Generate the evidence trajectories
for i = 1:num_trials
    [~, X_values(i, :)] = simple_model(B, sigma, dt, time_interval);
end

% Calculate the mean and variance over time
for t = 1:size(X_values, 2)
    mean_values(t) = mean(X_values(:, t));
    var_values(t) = var(X_values(:, t));
end

% Plot the trajectories, mean trajectory, and one standard deviation above and below the mean
figure;
hold on;
for i = 1:num_trials
    plot((1:size(X_values, 2)) * dt, X_values(i, :), 'Color', [0.7, 0.7, 0.7]);
end
plot((1:size(X_values, 2)) * dt, mean_values, 'k', 'LineWidth', 2);
plot((1:size(X_values, 2)) * dt, mean_values + sqrt(var_values), 'r', 'LineWidth', 2);
plot((1:size(X_values, 2)) * dt, mean_values - sqrt(var_values), 'r', 'LineWidth', 2);
hold off;
title('Evidence Trajectories');
xlabel('Time (s)');
ylabel('Evidence');

%% Q4
% Define the parameters
B = 0.1;
sigma = 1;
dt = 0.1;
time_interval = 10;
num_trials = 10;

% Initialize arrays to hold decision variable trajectories and means
X_values = zeros(num_trials, round(time_interval / dt));
mean_values = zeros(1, round(time_interval / dt));

% Generate the evidence trajectories
for i = 1:num_trials
    [~, X_values(i, :)] = simple_model(B, sigma, dt, time_interval);
end

% Calculate the mean and standard deviation over time
for t = 1:size(X_values, 2)
    mean_values(t) = mean(X_values(:, t));
end
simulation_std = std(X_values, 0, 1);

% Calculate the theoretical mean and standard deviation
t_values = (1:size(X_values, 2)) * dt;
theoretical_mean = B * t_values;
theoretical_std = sqrt(sigma^2 * t_values);

% Plot the decision variable, simulation mean, simulation std, theoretical mean, and theoretical std
figure;
hold on;
for i = 1:num_trials
    plot(t_values, X_values(i, :), 'Color', [0.6, 0.8, 0.6]); % Light Green
end
plot(t_values, mean_values, 'k', 'LineWidth', 2); % Black
plot(t_values, mean_values + simulation_std, 'Color', [1, 0, 0], 'LineWidth', 2); % Red
plot(t_values, mean_values - simulation_std, 'Color', [1, 0, 0], 'LineWidth', 2); % Red
plot(t_values, theoretical_mean, 'b--', 'LineWidth', 2); % Blue dashed line
plot(t_values, theoretical_mean + theoretical_std, 'm--', 'LineWidth', 2); % Magenta dashed line
plot(t_values, theoretical_mean - theoretical_std, 'm--', 'LineWidth', 2); % Magenta dashed line
hold off;
title('Decision Variable');
xlabel('Time (s)');
ylabel('Decision Variable');
legend('Trajectories', 'Simulation Mean', 'Simulation Std', 'Theoretical Mean', 'Theoretical Std', 'Location', 'northwest');

%%
% Set the parameters
theta_plus = 2;
theta_minus = -2;
sigma = 1;
x0 = 1;
bias = 0.1;

% Run multiple trials and collect evidence trajectories
num_trials = 10;
evidence_trajectories = zeros(num_trials, 1001); % 1001 time steps

for i = 1:num_trials
    [rt, ~] = two_choice_trial(theta_plus, theta_minus, sigma, x0, bias);
    dt = rt / 1000;
    t = 0:dt:rt;
    x = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt));
    evidence_trajectories(i, 1:numel(x)) = x;
end
% Plot the evidence trajectories
figure;
hold on;
time_limit = 10;
time_steps = size(evidence_trajectories, 2);
time_range = linspace(0, time_limit, time_steps);

for i = 1:num_trials
    x = evidence_trajectories(i, :);
    x(x > theta_plus) = theta_plus; % Clip values above theta_plus
    x(x < theta_minus) = theta_minus; % Clip values below theta_minus
    plot(time_range, x, 'Color', [0.7, 0.7, 0.7]);
end

plot([0, time_limit], [theta_plus, theta_plus], 'r--', 'LineWidth', 2);
plot([0, time_limit], [theta_minus, theta_minus], 'b--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Evidence');
title('Drift Diffusion Model - Evidence Trajectories');
hold off;



%% Q5

% Main script
% Set up the parameters for simple_model2
bias = 0.1;
time_limit = 10;
start_point = 0;

% Run simple_model2 and print the result
choice = simple_model2(bias, time_limit, start_point);
disp(['Choice (simple_model2): ', num2str(choice)])

% Set up the parameters for two_choice_trial
theta_positive = 1;
theta_negative = -1;
sigma = 1;
x0 = 0;
B = 0.1;

% Run two_choice_trial and print the results
[reaction_time, response] = two_choice_trial(theta_positive, theta_negative, sigma, x0, B);
disp(['Reaction Time: ', num2str(reaction_time), ', Response: ', num2str(response)])
%%
% Set the parameters
bias = 0.5;
time_limit = 1;
start_point = 0;

% Simulate a single trial
choice = simple_model2(bias, time_limit, start_point);

% Display the choice
disp(['Choice: ' num2str(choice)]);
%% Q5


% Set the parameters
bias = 0.5;
time_limit = 1;
start_point = 1;

% Simulate multiple trials
num_trials = 1000;
choices = zeros(1, num_trials);
for i = 1:num_trials
    choices(i) = simple_model2(bias, time_limit, start_point);
end

% Count the occurrences of each choice
unique_choices = unique(choices);
choice_counts = histcounts(choices, [unique_choices, unique_choices(end)+1]);

% Plot the distribution as a bar chart
figure;
bar(unique_choices, choice_counts);
xlabel('Choice');
ylabel('Count');
title('Distribution of Choices');

%% Q6




% Set the parameters
theta_plus = 2;
theta_minus = -2;
sigma = 1.5;
x0 = 1;
bias = 0.1;

% Run a single trial
[rt, response] = two_choice_trial(theta_plus, theta_minus, sigma, x0, bias);

% Display the results
disp("Reaction Time: " + rt + " seconds");
disp("Response: " + response);

% Plot the decision process
dt = 0.01; % Time step size for plotting
t = 0:dt:rt; % Time range
x = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt)); % Accumulated evidence over time

figure;
plot(t, x, 'LineWidth', 2);
hold on;
plot([0, rt], [theta_plus, theta_plus], 'r--', 'LineWidth', 2);
plot([0, rt], [theta_minus, theta_minus], 'b--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Evidence');
title('Drift Diffusion Model with Thresholds');
legend('Evidence', 'Positive Threshold', 'Negative Threshold');
hold off;


%% Q6

% Set the parameters
theta_plus = 2;
theta_minus = -2;
sigma = 1;
x0 = 1;
bias = 0.1;

% Run a single trial
[rt, response] = two_choice_trial(theta_plus, theta_minus, sigma, x0, bias);

% Display the results
disp("Reaction Time: " + rt + " seconds");
disp("Response: " + response);

% Plot the decision process
dt = 0.01; % Time step size for plotting
t = 0:dt:rt; % Time range
x = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt)); % Accumulated evidence over time

% Generate another evidence trajectory
x2 = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt)); % Accumulated evidence over time

figure;
plot(t, x, 'LineWidth', 2);
hold on;
plot(t, x2, 'Color', [0.3010, 0.7450, 0.9330], 'LineWidth', 2);
plot([0, rt], [theta_plus, theta_plus], 'r--', 'LineWidth', 2);
plot([0, rt], [theta_minus, theta_minus], 'b--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Evidence');
title('Drift Diffusion Model with Thresholds');
legend('Evidence 1', 'Evidence 2', 'Positive Threshold', 'Negative Threshold');
hold off;
 %% Q 6
 

% Set the parameters
theta_plus = 2;
theta_minus = -2;
sigma = 1;
x0 = 1;
bias = 0.1;

% Run multiple trials and collect evidence trajectories
num_trials = 7;
evidence_trajectories = zeros(num_trials, 1001); % 1001 time steps

for i = 1:num_trials
    [rt, ~] = two_choice_trial(theta_plus, theta_minus, sigma, x0, bias);
    dt = rt / 1000;
    t = 0:dt:rt;
    x = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt));
    evidence_trajectories(i, 1:numel(x)) = x;
end
% Plot the evidence trajectories
figure;
hold on;
time_limit = 10;
time_steps = size(evidence_trajectories, 2);
time_range = linspace(0, time_limit, time_steps);

for i = 1:num_trials
    x = evidence_trajectories(i, :);
    x(x > theta_plus) = theta_plus; % Clip values above theta_plus
    x(x < theta_minus) = theta_minus; % Clip values below theta_minus
    plot(time_range, x, 'Color', [0.7, 0.7, 0.7]);
end

plot([0, time_limit], [theta_plus, theta_plus], 'r--', 'LineWidth', 2);
plot([0, time_limit], [theta_minus, theta_minus], 'b--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Evidence');
title('Drift Diffusion Model - Evidence Trajectories');
hold off;



%% Q7 E
% Define the number of trials and parameters for the trials
num_trials = 100000;
theta_positive = 2;
theta_negative = -2;
sigma = 1;
x0 = 0;
B = 0.1;

% Initialize arrays to hold the reaction times and responses
reaction_times = zeros(1, num_trials);
responses = zeros(1, num_trials);

% Run the trials
for i = 1:num_trials
    [reaction_times(i), responses(i)] = two_choice_trial(theta_positive, theta_negative, sigma, x0, B);
end

% Determine the correctness of the responses
correct_responses = responses == sign(B);
incorrect_responses = ~correct_responses;

% Plot the reaction times
figure
hold on
scatter(reaction_times(correct_responses), ones(1, sum(correct_responses)), 'r')
scatter(reaction_times(incorrect_responses), -ones(1, sum(incorrect_responses)), 'b')
hold off

xlabel('Reaction Time')
ylabel('Correctness')
legend('Correct', 'Incorrect')
title('Reaction Time vs Correctness')
%% Q7
% Parameters
upperThreshold = 1.0;  
lowerThreshold = -1.0;
noiseLevel = 1.0;
initialEvidence =0.50;
positiveBias = 1.0;
negativeBias = -1.0;
timeStep = 0.1;
maxDuration = 100.0;

% Simulate race model
[selectedOption, responseTime] = simulate_race_model(upperThreshold, lowerThreshold, noiseLevel, initialEvidence, positiveBias, negativeBias, timeStep);

% Display results
fprintf('Race model: Choice %d made in %f seconds.\n', selectedOption, responseTime);

% Simulate race model with a fixed time interval
[chosenOption, reactionTime] = simulate_race_model_fixed_interval(upperThreshold, lowerThreshold, noiseLevel, initialEvidence, positiveBias, negativeBias, timeStep, maxDuration);

% Display results
fprintf('Race model with fixed interval: Choice %d made in %f seconds.\n', chosenOption, reactionTime);

%% Q8 


% Set the parameters
theta_plus = 2;
theta_minus = -2;
sigma = 1;
x0 = 0;
bias = 0.1;
max_time = 5;

% Run the race trial
[best_choice, reaction_time] = race_trial(theta_plus, theta_minus, sigma, x0, bias, max_time);

% Plot the result
figure;
hold on;

% Plot the evidence trajectory
dt = 0.01; % Time step size for plotting
t = 0:dt:max_time; % Time range
x = x0 + cumsum(bias * dt + sigma * randn(size(t)) * sqrt(dt)); % Accumulated evidence over time
plot(t, x, 'LineWidth', 2);

% Plot the thresholds
plot([0, max_time], [theta_plus, theta_plus], 'r--', 'LineWidth', 2);
plot([0, max_time], [theta_minus, theta_minus], 'b--', 'LineWidth', 2);

% Plot the reaction time
if reaction_time < max_time
    plot(reaction_time, x(find(t==reaction_time)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
else
    plot(max_time, x(end), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
end

% Set labels and title
xlabel('Time (s)');
ylabel('Evidence');
title('Race Model - Decision Process');

% Set legend
legend('Evidence', 'Positive Threshold', 'Negative Threshold', 'Reaction Time');

hold off;

%%
% Set the parameters
start_point1 = 0.5;
start_point2 = 0;
bias1 = 0.05;
bias2 = 0.05;
sigma1 = 1;
sigma2 = 1;
threshold1 = 2;
threshold2 = 2;
dt = 0.1;
max_time = 5;

% Run the race trial
[best_choice, reaction_time] = race_trial(start_point1, start_point2, sigma1, sigma2, threshold1, threshold2, bias1, bias2, dt, max_time);

% Plot the result
figure;
hold on;

% Plot the evidence trajectory
t = 0:dt:max_time; % Time range
x1 = start_point1 + cumsum(bias1 * dt + sigma1 * randn(size(t)) * sqrt(dt)); % Accumulated evidence for choice 1
x2 = start_point2 + cumsum(bias2 * dt + sigma2 * randn(size(t)) * sqrt(dt)); % Accumulated evidence for choice 2
plot(t, x1, 'LineWidth', 2, 'Color', 'b');
plot(t, x2, 'LineWidth', 2, 'Color', 'r');

% Plot the thresholds
plot([0, max_time], [threshold1, threshold1], 'r--', 'LineWidth', 2);
plot([0, max_time], [threshold2, threshold2], 'b--', 'LineWidth', 2);

% Plot the reaction time
if reaction_time < max_time
    if best_choice == 1
        plot(reaction_time, x1(find(t==reaction_time)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    else
        plot(reaction_time, x2(find(t==reaction_time)), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    end
else
    if best_choice == 1
        plot(max_time, x1(end), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    else
        plot(max_time, x2(end), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    end
end

% Set labels and title
xlabel('Time (s)');
ylabel('Evidence');
title('Race Model - Decision Process');

% Set legend
legend('Choice 1 Evidence', 'Choice 2 Evidence', 'Threshold 1', 'Threshold 2', 'Reaction Time');

hold off;



%% PART 2 
% Generate a random sequence of stimulus orientations
num_steps = 1000;
stimulus = rand(1, num_steps)*180 - 90;  % Random orientations between -90 and 90

% Initialize parameters
MT_p_values = [0.5, 0.5];  % Firing probabilities for the MT neurons
LIP_weights = [1, -1];  % Weighting factors for the MT neurons
LIP_threshold = 50;  % LIP firing rate that represents the choice threshold criterion
Evidence_thr = 0.5;  % Threshold for the evidence

% Run the simulation
[MT_neurons, LIP_neurons] = simulate_MT_LIP_neurons(stimulus, MT_p_values, LIP_weights, LIP_threshold, Evidence_thr);

% Plot the activity patterns for each neuron
figure
for neuron = 1:2
    subplot(2, 2, neuron)
    rasterplot(MT_neurons{neuron,:}, 'k')
    title(['MT Neuron ' num2str(neuron)])
    
    subplot(2, 2, neuron+2)
    rasterplot(LIP_neurons{neuron,:}, 'k')
    title(['LIP Neuron ' num2str(neuron)])
end
% 



% 
% % Question 1:
% clear; clc; close all;
% MT_p_values = [0.07 0.04];
% LIP_weights = [0.1 -0.15];
% LIP_threshold = 30;
% [RT, MT1_event_times, MT2_event_times, LIP_event_times] = lip_activity(...
%     MT_p_values, LIP_weights, LIP_threshold)
% 
% figure;
% hold on;
% for i = 1: length(MT1_event_times)
%     plot([MT1_event_times(i) MT1_event_times(i)], [1 1.5], 'color', 'k', 'linewidth', 1.5); 
% end
% for i = 1: length(MT2_event_times)
%     plot([MT2_event_times(i) MT2_event_times(i)], [1.5 2], 'color', [0.6 0.6 0.6],  'linewidth', 1.5); 
% end
% for i = 1: length(LIP_event_times)
%     plot([LIP_event_times(i) LIP_event_times(i)], [2 2.5], 'color', [0.3010 0.7450 0.9330], 'linewidth', 1.5); 
% end
% legend('M1', 'M2', 'LIP');
% ylim([0.5 3]);
% xlabel('time (sec)'); title('Raster plot');
% 
% % Question 2:
% clear; clc; close all;    
% 
% LIP1_weights = [0.1 -0.1];
% LIP2_weights = [-0.1 0.1];
% 
% MT_p_values_All = [0.05*ones(1, 250), 0.05*ones(1, 250), 0.05*ones(1, 250), 0.05*ones(1, 250);...
%     0.08*ones(1, 250), 0.02*ones(1, 250), 0.06*ones(1, 250), 0.03*ones(1, 250)];
%      
% [MT1_event_times, MT2_event_times, LIP1_event_times, LIP2_event_times] = ...
%     lip_activity2(MT_p_values_All, LIP1_weights, LIP2_weights);
% 
% figure;
% hold on;
% for i = 1: length(MT1_event_times)
%     plot([MT1_event_times(i) MT1_event_times(i)], [1 1.5], 'color', 'k', 'linewidth', 1.5); 
% end
% for i = 1: length(MT2_event_times)
%     plot([MT2_event_times(i) MT2_event_times(i)], [1.5 2], 'color', [0.6 0.6 0.6],  'linewidth', 1.5); 
% end
% for i = 1: length(LIP1_event_times)
%     plot([LIP1_event_times(i) LIP1_event_times(i)], [2 2.5], 'color', [0.3010 0.7450 0.9330], 'linewidth', 1.5); 
% end
% for i = 1: length(LIP2_event_times)
%     plot([LIP2_event_times(i) LIP2_event_times(i)], [2.5 3], 'color', [0 0.4470 0.7410], 'linewidth', 1.5); 
% end
% legend('M1', 'M2', 'LIP1', 'LIP2');
% xline(0.25, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off'); hold all;
% xline(0.5, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off');
% xline(0.75, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off');
% xlabel('time (sec)'); title('Raster plot');
% 

%%

% Question 1:
clc; clear; close all;

MT_p_values = [0.07 0.04];
LIP_weights = [0.1 -0.15];
LIP_threshold = 30;

[LIP_event_times] = simulate_LIP_activity(MT_p_values, LIP_weights, LIP_threshold);

figure;
hold on;
for i = 1:length(LIP_event_times)
    plot([LIP_event_times(i) LIP_event_times(i)], [0 1], 'color', [0.3010 0.7450 0.9330], 'linewidth', 1.2);
end

ylim([0 1]);
xlabel('Time (sec)');
ylabel('Neuron');
title('Raster Plot - LIP Neuron');
set(gca, 'ytick', []);
box on;

% Question 2:
clc; clear; close all;

LIP1_weights = [0.1 -0.1];
LIP2_weights = [-0.1 0.1];

MT_p_values_All = [0.05 * ones(1, 250), 0.05 * ones(1, 250), 0.05 * ones(1, 250), 0.05 * ones(1, 250);...
                   0.08 * ones(1, 250), 0.02 * ones(1, 250), 0.06 * ones(1, 250), 0.03 * ones(1, 250)];

[MT1_event_times, MT2_event_times, LIP1_event_times, LIP2_event_times] = ...
    simulate_MT_LIP_neurons(MT_p_values_All, LIP1_weights, LIP2_weights);



figure;
hold on;
for i = 1:length(LIP1_event_times)
    plot([LIP1_event_times(i) LIP1_event_times(i)], [1 2], 'color', [0.3010 0.7450 0.9330], 'linewidth', 1.2);
end

for i = 1:length(LIP2_event_times)
    plot([LIP2_event_times(i) LIP2_event_times(i)], [2.5 3.5], 'color', [0 0.4470 0.7410], 'linewidth', 1.2);
end

for i = 1:length(MT1_event_times)
    plot([MT1_event_times(i) MT1_event_times(i)], [4 5], 'color', 'k', 'linewidth', 1.2);
end

for i = 1:length(MT2_event_times)
    plot([MT2_event_times(i) MT2_event_times(i)], [5.5 6.5], 'color', [0.6 0.6 0.6], 'linewidth', 1.2);
end

legend('LIP1', 'LIP2', 'MT1', 'MT2');
xline(0.25, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off');
xline(0.5, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off');
xline(0.75, '--', 'color', 'r', 'linewidth', 1.3, 'handlevisibility', 'off');
xlabel('Time (sec)');
ylabel('Neuron');
title('Raster Plot - MT and LIP Neurons');
set(gca, 'ytick', []);
box on;




















