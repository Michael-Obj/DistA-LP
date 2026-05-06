function server = creat_destination(server, env_parameters)
    nr_loc = size(env_parameters.latitude, 1); 
    server.destination_loc_ID = randsample(nr_loc, server.nr_destination);
end