
function [RT, MT1_event_times, MT2_event_times, LIP_event_times] = lip_activity(...
    MT_p_values, LIP_weights, LIP_threshold)

    % Parameters:
    % MT_p_values - a vector with 2 elements, firing probabilities for the
    % excitatory and inhibitory neurons, resp.
    % LIP_weights - a length 2 vector of weighting factors for the evidence
    % from the excitatory (positive) and inhibitory (negative) neurons
    % LIP_threshold - the LIP firing rate that represents the choice threshold criterion
    % use fixed time scale of 1 ms
    
    t = 0;
    dt = 0.001;
    N = [0 0]; % plus is first, minus is second
    rate = 0.0;
    LIP_event_times = [];
    MT1_event_times = [];
    MT2_event_times = [];
    M = 100;
    rate = 0;
    
    while rate < LIP_threshold
        
        t = t + dt;
        
        dN = rand(1, 2) < MT_p_values;
        
        if dN(1) == 1, MT1_event_times = [MT1_event_times, t]; end
        if dN(2) == 1, MT2_event_times = [MT2_event_times, t]; end
        
        N = N + dN;
        
        p_LIP = sum(N .* LIP_weights);
        LIP_event = rand < p_LIP;
        
        if LIP_event, LIP_event_times = [LIP_event_times t]; end
        
        
        % check LIP mean rate for last M spikes
        if length(LIP_event_times) > M
            rate = M/(t - LIP_event_times(length(LIP_event_times) - M + 1));
        end
        
    end
    RT = t;
end
