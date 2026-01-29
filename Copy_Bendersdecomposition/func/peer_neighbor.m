function peerNeighbor = peer_neighbor(G, peerMatrix, NR_LOC)
    peerNeighbor = sparse(NR_LOC, NR_LOC); 
    for i = 1:1:NR_LOC
        peer_vector = find(peerMatrix(i, :)>0); 
        for j = 1:1:size(peer_vector, 2)
            P = shortestpath(G, i, peer_vector(1, j)); 
            for l = 2:1:size(P, 2)
                if any(peer_vector == P(1, l))
                    peerNeighbor(i, P(1, l)) = 1;
                    break; 
                end
            end
        end
    end

    
end