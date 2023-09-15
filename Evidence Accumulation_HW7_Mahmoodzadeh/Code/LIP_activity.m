
function [LIP_event_times, MT_event_times] = LIP_activity(MT_p_values, LIP_weights, LIP_threshold, Evidence_thr)
    dt=0.001;
    rate=0;
    N=[0 0]; 
    t=0;
    LIP_event_times=[];
    MT_event_times={[],[]};
    
    while rate<LIP_threshold
        dN = rand(1,2) < MT_p_values;
        N = N + dN;

        for i=1:2
            if dN(i)
                MT_event_times{i} = [MT_event_times{i} t];
            end
        end

        p_lip = sum(N.*LIP_weights);
        LIP_event = Evidence_thr < p_lip;

        if LIP_event == 1
            LIP_event_times = [LIP_event_times t];
        end

        M = 100;
        if length(LIP_event_times)>=M
            rate = M/(t-LIP_event_times(end-M+1));
        end
        t=t+dt;
    end
end
