function choice = simple_model2(bias, time_limit, start_point)
    % Calculate the probability of being above the start point
    p_above = (1 + erf((start_point - bias) / sqrt(2))) / 2;
    
    % Generate a random number
    random_num = rand;
    
    % Make a choice based on the random number and the probability
    if random_num <= p_above
        choice = 1; % Above the start point
    else
        choice = -1; % Below the start point
    end
end


