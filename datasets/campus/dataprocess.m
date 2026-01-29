%% Process the campus map information

nodelist_process = xlsread('./dataprocess/nodelist1_process.xlsx');
edgelist_process = xlsread('./dataprocess/edgelist_process.xlsx'); 


node_ID = nodelist_process(:, 1);
uniqueVals = unique(node_ID);



valCount = hist(node_ID, uniqueVals )';


for i = 1:1:size(edgelist_process, 1)
    start_point = find(node_ID == edgelist_process(i, 1))-1; 
    end_point = find(node_ID == edgelist_process(i, 2))-1; 
    edgelist(i, :) = [start_point, end_point]; 
end
    edgelist2 = [edgelist(:, 2), edgelist(:, 1)];
    edgelist = [edgelist; edgelist2]; 
a = 0; 