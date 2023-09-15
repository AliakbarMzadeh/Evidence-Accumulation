function [best_choice, reaction_time] = race_trial(theta_plus, theta_minus, sigma, x0, bias, max_time)
    dt = 0.1; % Time step size
    t = 0; % Initialize time
    x = x0; % Initialize evidence variable

    % Simulate the decision process until a threshold is crossed or maximum time is reached
    while t < max_time
        % Update evidence variable
        dx = bias * dt + sigma * randn * sqrt(dt);
        x = x + dx;

        % Check if thresholds are crossed
        if x >= theta_plus
            best_choice = 1; % Positive choice
            reaction_time = t; % Reaction time is the current time
            return; % Exit the function
        elseif x <= theta_minus
            best_choice = -1; % Negative choice
            reaction_time = t; % Reaction time is the current time
            return; % Exit the function
        end

        t = t + dt; % Increment time
    end

    % If maximum time is reached without a clear winner, choose the best option based on the accumulated evidence
    if x >= 0
        best_choice = 1; % Positive choice
    else
        best_choice = -1; % Negative choice
    end
    reaction_time = max_time; % Set the reaction time to the maximum time
end

% Modify the simple_model function to also return decision variable trajectory
function [choice, X] = simple_model(B, sigma, dt, time_interval)
    num_steps = round(time_interval / dt);
    x0 = 0;
    X = zeros(1, num_steps);
    X(1) = x0;
    for t = 2:num_steps
        dW = sqrt(dt) * randn();
        dX = B * dt + sigma * dW;
        X(t) = X(t-1) + dX;
    end
    choice = X(end) > 0;
end
