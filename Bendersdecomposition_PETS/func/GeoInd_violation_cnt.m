function [nr_violations, violation_mag] = GeoInd_violation_cnt(user, env_parameters)
    nr_violations = 0; 
    violation_mag = [];
    for i = 1:1:size(user, 1)
        i
        for j = i+1:1:size(user, 1)
            obf_matrix_i = ones(env_parameters.nr_loc_selected, env_parameters.nr_loc_selected)*0.000; 
            obf_matrix_i(user(i, 1).LR_loc_ID, user(i, 1).obf_loc_ID) = user(i, 1).obfuscation_matrix; 
            obf_matrix_j = ones(env_parameters.nr_loc_selected, env_parameters.nr_loc_selected)*0.000;
            obf_matrix_j(user(j, 1).LR_loc_ID, user(j, 1).obf_loc_ID) = user(j, 1).obfuscation_matrix;
            for idx_i = 1:1:env_parameters.nr_loc_selected 
                for idx_j = 1:1:env_parameters.nr_loc_selected
                    loc1 = [env_parameters.longitude_selected(idx_i), env_parameters.latitude_selected(idx_i)]; 
                    loc2 = [env_parameters.longitude_selected(idx_j), env_parameters.latitude_selected(idx_j)]; 
                    [distance, ~, ~] = haversine(loc1, loc2); 
                    for k = 1:1:env_parameters.nr_loc_selected
                        if obf_matrix_i(idx_i, k) > exp(distance*env_parameters.EPSILON)*obf_matrix_i(idx_j, k)
                            nr_violations = nr_violations + 1;
                            violation_mag = [violation_mag obf_matrix_i(idx_i, k)-(exp(distance*env_parameters.EPSILON)*obf_matrix_i(idx_j, k))]; 
                        end
                    end
                end
            end
        end
    end
    nr_violations = nr_violations/env_parameters.nr_loc_selected^3/(size(user, 1)*(size(user, 1)-1));
    violation_mag = mean(violation_mag); 
end