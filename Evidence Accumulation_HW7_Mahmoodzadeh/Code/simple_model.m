function choice = simple_model(bias, sigma, dt, time_interval)
    % Generate evidence trajectory
    t = 0:dt:time_interval;
    evidence = bias * t + sigma * randn(size(t)) * sqrt(dt);
    
    % Calculate the accumulated evidence
    accumulated_evidence = sum(evidence);
    
    % Make the choice based on accumulated evidence
    if accumulated_evidence > 0
        choice = 1; % Go response
    else
        choice = -1; % No Go response
    end
end
