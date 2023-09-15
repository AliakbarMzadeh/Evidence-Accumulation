


% For Question 7:

function [selectedOption, responseTime] = simulate_race_model(upperLimit, lowerLimit, noiseScale, initialEvidence, positiveBias, negativeBias, timeStep)
    % Initial evidence accumulation for both options
    accumulatedEvidence_positive = initialEvidence;
    accumulatedEvidence_negative = initialEvidence;

    % Initiate time 
    currentTime = 0;

    % Loop until one evidence crosses its threshold
    while (accumulatedEvidence_positive < upperLimit && accumulatedEvidence_negative < lowerLimit)
        randomNoise = sqrt(timeStep) * randn; % Gaussian noise (Brownian motion)
        accumulatedEvidence_positive = accumulatedEvidence_positive + positiveBias * timeStep + noiseScale * randomNoise;
        accumulatedEvidence_negative = accumulatedEvidence_negative + negativeBias * timeStep + noiseScale * randomNoise;

        currentTime = currentTime + timeStep;
    end

    % Determine the winning choice
    if accumulatedEvidence_positive >= upperLimit
        selectedOption = 1;
    else
        selectedOption = -1;
    end

    responseTime = currentTime;
end

% For Question 8:

function [chosenOption, reactionTime] = simulate_race_model_fixed_interval(upperThreshold, lowerThreshold, errorVariance, initialAccumulation, positivePropensity, negativePropensity, deltaT, maxDuration)
    % Initial accumulation of evidence for both choices
    evidence_positive = initialAccumulation;
    evidence_negative = initialAccumulation;

    % Time initialization
    currentTime = 0;

    % Loop until one evidence crosses its threshold or maximum duration is reached
    while (evidence_positive < upperThreshold && evidence_negative < lowerThreshold && currentTime < maxDuration)
        gaussianNoise = sqrt(deltaT) * randn; % Gaussian noise (Brownian motion)
        evidence_positive = evidence_positive + positivePropensity * deltaT + errorVariance * gaussianNoise;
        evidence_negative = evidence_negative + negativePropensity * deltaT + errorVariance * gaussianNoise;

        currentTime = currentTime + deltaT;
    end

    % Determine the winning choice, or pick the choice with more evidence if time has run out
    if evidence_positive >= upperThreshold
        chosenOption = 1;
    elseif evidence_negative >= lowerThreshold
        chosenOption = -1;
    elseif evidence_positive > evidence_negative
        chosenOption = 1;
    else
        chosenOption = -1;
    end

    reactionTime = currentTime;
end




function [MT_neurons, LIP_neurons] = simulate_MT_LIP_neurons(stimulus, MT_p_values, LIP_weights, LIP_threshold, Evidence_thr)
    num_steps = length(stimulus);
    MT_neurons = cell(2, num_steps);
    LIP_neurons = cell(2, num_steps);
    
    for step = 1:num_steps
        orientation = stimulus(step);
        
        % Change the firing probabilities based on the orientation of the stimulus
        MT_p_values(1) = calculate_MT_p_value(orientation);
        MT_p_values(2) = calculate_MT_p_value(-orientation);
        
        % Generate activity patterns for each MT and LIP neuron
        for neuron = 1:2
            [LIP_event_times, MT_event_times] = LIP_activity(MT_p_values, LIP_weights, LIP_threshold, Evidence_thr);
            
            MT_neurons{neuron, step} = MT_event_times{neuron};
            LIP_neurons{neuron, step} = LIP_event_times;
        end
    end
end

function p_value = calculate_MT_p_value(orientation)
    % Perform calculations based on the stimulus orientation to determine the firing probability for MT neurons
    % Replace the following example code with your specific calculations
    if orientation > 0
        p_value = 0.8;
    else
        p_value = 0.2;
    end
end

% Final part
function rasterplot(event_times, color)
    for i = 1:length(event_times)
        for j = 1:length(event_times{i})
            line([event_times{i}(j) event_times{i}(j)], [i-1 i], 'Color', color);
        end
    end
    xlim([0 max(cellfun(@max, event_times))])
    ylim([0 length(event_times)])
end







% Function to simulate the activity of a single LIP neuron with fixed probabilities
% for its corresponding excitatory and inhibitory MT neurons
function [LIP_event_times] = simulate_LIP_activity(MT_p_values, LIP_weights, LIP_threshold)
    % Parameters:
    % MT_p_values - a vector with 2 elements, firing probabilities for the
    % excitatory and inhibitory neurons, respectively.
    % LIP_weights - a length 2 vector of weighting factors for the evidence
    % from the excitatory (positive) and inhibitory (negative) neurons.
    % LIP_threshold - the LIP firing rate that represents the choice threshold criterion.
    
    dt = 0.001; % Time step in seconds
    rate = 0;
    N = [0 0]; % plus is first, minus is second
    t = 0;
    LIP_event_times = [];
    
    while rate < LIP_threshold
        dN = rand(1, 2) < MT_p_values;
        N = N + dN;

        p_lip = sum(N .* LIP_weights);
        LIP_event = rand < p_lip;

        if LIP_event
            LIP_event_times = [LIP_event_times t];
        end

        % Check LIP mean rate for last M spikes
        M = 100;
        if length(LIP_event_times) >= M
            rate = M / (t - LIP_event_times(end - M + 1));
        end
        
        t = t + dt;
    end
end
