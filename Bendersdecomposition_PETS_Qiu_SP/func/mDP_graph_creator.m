function G_mDP = mDP_graph_creator(env_parameters)
    s = []; 
    t = [];
    weights = []; 
    for i = 1:1:env_parameters.nr_loc_selected
        for j = i+1:1:env_parameters.nr_loc_selected
            loc1 = [env_parameters.longitude_selected(i, 1), env_parameters.latitude_selected(i, 1)];
            loc2 = [env_parameters.longitude_selected(j, 1), env_parameters.latitude_selected(j, 1)];
            [distance_inst, ~, ~] = haversine(loc1, loc2);
            if distance_inst <= env_parameters.NEIGHBOR_THRESHOLD
                s = [s; i; j]; 
                t = [t; j; i]; 
                weights = [weights; distance_inst; distance_inst];
            end
        end
    end
    G_mDP = graph(s, t, weights); 
end