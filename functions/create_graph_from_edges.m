% Function to create a graph from the edges data
function G = create_graph_from_edges(df_edges, df_nodes)
    % Assuming 'df_edges' contains columns 'u', 'v', and 'length'
    
    id_list = df_nodes.osmid;
    
    edges_u = int64(df_edges.u);
    edges_v = int64(df_edges.v);
    edges_weight = df_edges.length;

    % Create graph object with weighted edges
    edges_u_index = zeros(size(edges_u));
    edges_v_index = zeros(size(edges_v));
    
    for i = 1:numel(edges_u)
        edges_u_index(i) = find(id_list == edges_u(i));
        edges_v_index(i) = find(id_list == edges_v(i));
    end
    
    G = graph(edges_u_index, edges_v_index, edges_weight);
end