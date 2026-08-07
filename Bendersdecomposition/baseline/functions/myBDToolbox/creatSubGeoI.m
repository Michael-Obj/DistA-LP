function agent = creatSubGeoI(agent, distance_matrix, NR_OBFLOC, EPSILON, beta, xi_hathat)
    distance_matrix = full(distance_matrix); 
    idx = 0; 

    NR_NODE_INTRASET = size(agent.node_intraset, 2); 
    NR_NODE_INTERSECT = size(agent.node_interset, 2); 

    agent.GeoI = sparse(1, (NR_NODE_INTRASET+NR_NODE_INTERSECT)*NR_OBFLOC);
    for i = 1:1:size(agent.intraset, 1)
        node_i = find(agent.node_intraset == agent.intraset(i,1)); 
        node_j = find(agent.node_intraset == agent.intraset(i,2)); 
        if size(node_i, 2)==0 
            % if node_i is in the inter set and node_j is in the intra set
            node_i = NR_NODE_INTRASET + find(agent.node_interset == agent.intraset(i,1));
        end

        if size(node_j, 2)==0
            % if node_i is in the inter set and node_j is in the intra set
            node_j = NR_NODE_INTRASET + find(agent.node_interset == agent.intraset(i,2));
        end

        for k = 1:1:NR_OBFLOC
            if node_i ~= node_j
                agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = 1;
                max_di=max(EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)),0);
                agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = -exp(max_di*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_hathat(agent.intraset(i,1), agent.intraset(i,2)));
                agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k)=min(agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k),-1);
                %agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = -exp(EPSILON*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-log(beta(agent.intraset(i,1), agent.intraset(i,2))));
                % if ((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_max)/(EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))>0.5
                %     agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = -exp((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_max);
                % else
                %     agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = -exp((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))/2);
                % end
                idx = idx + 1;
                max_di=max(EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)),0);
                agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = -exp(max_di*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_hathat(agent.intraset(i,1), agent.intraset(i,2)));
                agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k)=min(agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k),-1);
                %agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = -exp(EPSILON*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-log(beta(agent.intraset(i,1), agent.intraset(i,2))));
                % if ((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_max)/(EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))>0.5
                %     agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = -exp((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))-xi_max);
                % else
                %     agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = -exp((EPSILON-beta(agent.intraset(i,1), agent.intraset(i,2)))*distance_matrix(agent.intraset(i,1), agent.intraset(i,2))/2);
                % end
                agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = 1;
                idx = idx + 1;
            end
        end
    end

    % for i = 1:1:size(agent.interset, 1)
    %     for k = 1:1:NR_OBFLOC
    %         node_i = find(agent.node == agent.intraset(i,1)); 
    %         node_j = NR_NODE_INTRASET + find(agent.node_interset == agent.interset(i,2)); 
    %         agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = 1;
    %         agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = -exp(EPSILON*distance_matrix(agent.interset(i,1), agent.interset(i,2)));
    %         idx = idx + 1;
    %         agent.GeoI(idx+1, (node_i-1)*NR_OBFLOC + k) = -exp(EPSILON*distance_matrix(agent.interset(i,1), agent.interset(i,2)));
    %         agent.GeoI(idx+1, (node_j-1)*NR_OBFLOC + k) = 1;
    %         idx = idx + 1;
    %     end
    % end
end