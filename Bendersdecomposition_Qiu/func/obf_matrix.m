function [z, z_fval] = obf_matrix(x_coord, y_coord, CPR_prior_prob, patient_index)
    % NR_LOC: Number of discrete locations
    
    % x_coord: the vector including the x coordinate of all the discrete locations
    % y_coord: the vector including the y coordinate of all the discrete locations
    % CPR_prior_prob: the prior distribution of workers 
    
    % Represent obfuscation matrix Z by a vector z, where each entry z_{l,k} in Z 
    % is written as z((l-1)*NR_PATIENT + k) in the vector
   
    parameters; 
    
    %% Inequality constraint A, b
    % Geo-indistinguishability constraints: 
    % the number of constraints is NR_LOC*(NR_LOC-1)*NR_LOC
    % the number of decision variables is NR_LOC*NR_LOC
     
	A = zeros(NR_LOC*(NR_LOC-1)*NR_LOC, NR_LOC*NR_LOC);
    
    row_indx = 1;
    for i = 1:1:NR_LOC
        for j = i+1:1:NR_LOC
            for k = 1:1:NR_LOC
                distance = sqrt((x_coord(1, i) - x_coord(1, j))^2 + (y_coord(1, i) - y_coord(1, j))^2); % distance between location i and location j
                A(row_indx, (i-1)*NR_LOC+k) = 1;               
                A(row_indx, (j-1)*NR_LOC+k) = -exp(EPSILON*distance);
                % z_{i,k} - z_{j,k}*exp(EPSILON*distance) <= 0
                row_indx = row_indx + 1;
                
                A(row_indx, (j-1)*NR_LOC+k) = 1;
                A(row_indx, (i-1)*NR_LOC+k) = -exp(EPSILON*distance);
                % z_{j,k} - z_{i,k}*exp(EPSILON*distance) <= 0
                row_indx = row_indx + 1;
            end
        end
    end
    b = zeros(NR_LOC*(NR_LOC-1)*NR_LOC, 1); 
    
%     A = [];
%     b = [];
    
    %% Equality constraint Aeq, beq

    % Probablity unit measure
	Aeq = zeros(NR_LOC, NR_LOC*NR_LOC);
    beq = ones(NR_LOC, 1); 
    
    for k = 1:1:NR_LOC
        for l = 1:1:NR_LOC
            Aeq(k, (k-1)*NR_LOC + l) =1;
        end
        % z_{k, 1}+z_{k, 2}+...+z_{k, K} = 1
    end
    
    %% Each z is in the range of [0, 1]
	lb = zeros(NR_LOC*NR_LOC,1);
    ub = ones(NR_LOC*NR_LOC,1);
    
    %% Objective function: 
    % objective: sum_{k,l} sum_n CPR_prior_prob(k, 1)*abs(distance_{obfuscated location l to destination n} - distance__{real location k to destination n}) z_{k,l}
    % => the coefficint for each z_{k,l} is: sum_n CPR_prior_prob(k, 1)*abs(distance_{obfuscated location l to destination n} - distance__{real location k to destination n})
    f = zeros(NR_LOC*NR_LOC, 1);
	for k = 1:1:NR_LOC
        for l = 1:1:NR_LOC
            f((k-1)*NR_LOC + l, 1) = 0;
            for j = 1:1:NR_PATIENT
                % calculate the distance from the obfuscated location l to the patient
                distance_l = sqrt((x_coord(l) - x_coord(patient_index(j)))^2 + (y_coord(l) - y_coord(patient_index(j)))^2);
                % calculate the distance from the real location k to the patient
                distance_k = sqrt((x_coord(k) - x_coord(patient_index(j)))^2 + (y_coord(k) - y_coord(patient_index(j)))^2);
                % add the distance error to the coefficient for z_{k, l}
                f((k-1)*NR_LOC + l, 1) = f((k-1)*NR_LOC + l, 1) + abs(distance_l - distance_k); 
            end
            f((k-1)*NR_LOC + l, 1) = f((k-1)*NR_LOC + l, 1)*CPR_prior_prob(k, 1); 
        end     
    end
    options = optimoptions('linprog','Algorithm','dual-simplex');


    [z, z_fval] = linprog(f, A, b, Aeq, beq, lb, ub, options);                       % Represent obfuscation matrix Z by a vector z, where each z_{i,j} -> z((i-1)*NR_LOC + j)
    z = reshape(z, NR_LOC, NR_LOC);                                         % reshape the obfuscation vector to matrix
    z = z'; 
    f_ = reshape(f, NR_LOC, NR_LOC); 
    f_ = f_';
end

