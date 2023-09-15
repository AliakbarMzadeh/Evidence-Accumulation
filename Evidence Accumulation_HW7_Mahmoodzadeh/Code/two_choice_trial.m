

function [rt, response] = two_choice_trial(theta_plus, theta_minus, sigma, x0, bias)
    dt = 0.1; % Time step size
    t = 0; % Initialize time
    x = x0; % Initialize evidence variable

    % Simulate the decision process until a threshold is crossed
    while true
        % Update evidence variable
        dx = bias * dt + sigma * randn * sqrt(dt);
        x = x + dx;

        % Check if threshold is crossed
        if x >= theta_plus
            rt = t; % Reaction time is the current time
            response = 1; % Positive response
            break;
        elseif x <= theta_minus
            rt = t; % Reaction time is the current time
            response = -1; % Negative response
            break;
        end

        t = t + dt; % Increment time
    end
end
