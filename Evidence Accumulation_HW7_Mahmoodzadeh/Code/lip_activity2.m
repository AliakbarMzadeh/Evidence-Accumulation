
function [MT1_event_times, MT2_event_times, LIP1_event_times, LIP2_event_times] = ...
    lip_activity2(MT_p_values_All, LIP1_weights, LIP2_weights)
    
    t = 0;
    dt = 0.001;
    N = [0 0]; % plus is first, minus is second
    rate = 0.0;
    LIP1_event_times = [];
    LIP2_event_times = [];
    MT1_event_times = [];
    MT2_event_times = [];
    
    M = 100;
    rate = [0 0];   %rate LIP1, rate LIP2
    
    while t < size(MT_p_values_All, 2)/1000
        
        t = t + dt;
        
        MT_p_values = MT_p_values_All(:, round(t*1000))';
        
        dN = rand(1, 2) < MT_p_values;
        
        if dN(1) == 1, MT1_event_times = [MT1_event_times, t]; end
        if dN(2) == 1, MT2_event_times = [MT2_event_times, t]; end
        
        N = N + dN;
        
        p_LIP1 = sum(N .* LIP1_weights);
        p_LIP2 = sum(N .* LIP2_weights);
        
        LIP_event1 = rand < p_LIP1;
        LIP_event2 = rand < p_LIP2;
        
        if LIP_event1, LIP1_event_times = [LIP1_event_times t]; end
        if LIP_event2, LIP2_event_times = [LIP2_event_times t]; end
        
    end
end
