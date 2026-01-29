function[G, edges_u_index, edges_v_index, edges_weight] = graph_preparation(df_nodes, df_edges, target_region)
%% Description:
    % The graph_preparation function is create the graph from the
    % openstreet dataset, where the edges weight will be travel time for
    % that edge.
%% Input:
    % df_nodes: Openstreet map nodes details
    % df_edges: Openstreet map edges details

%% Output
    % G: The resultant graph
    % edges_u_index: Since the nodes ids has been changed to index,
    % therefore startpoint ids indices are in edges_u_index
    % edges_v_index: endpoint ids indices
    % timeTaken: travel time for each edge
    
%%    

    id_list = df_nodes.osmid;
    

    % convert to int
    edges_u = int64(df_edges.u);
    edges_v = int64(df_edges.v);
    edges_weight = df_edges.length;
    % timeTaken = (df_edges.time);


    % Create a graph object with weighted edges

    % First, translate the node index to the range [1, 2, ...]

    edges_u_index = zeros(size(edges_u));
    edges_v_index = zeros(size(edges_v));
    index = 1; 
    for i = 1:1:size(edges_u, 1)
        if size(find(id_list == edges_u(i)), 1) > 0 & size(find(id_list == edges_v(i)), 1) > 0
            edges_u_index(index, 1) = find(id_list == edges_u(i));
            edges_v_index(index, 1) = find(id_list == edges_v(i));
            index = index + 1;
        end
    end
    
    % G_original = graph(edges_u_index, edges_v_index, timeTaken);
    G_original = graph([edges_u_index; edges_v_index], [edges_v_index; edges_u_index], [edges_weight; edges_weight]);

    % G = subgraph(G_original, target_region); 
    G = G_original; 

    % % Plot the graph with node locations
    %  figure;
    %  p = plot(G, 'XData', lon, 'YData', lat, 'EdgeLabel', G.Edges.Weight);
    %  title('Weighted Graph');
end