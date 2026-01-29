function draw3DBar(matrixData)
% DRAW3DBAR - Draws a 3D bar plot from a 2D matrix
%
% Syntax: draw3DBar(matrixData)
%
% Input:
%   matrixData - A 2D numeric matrix whose values determine the bar heights

    % Check input
    if ~ismatrix(matrixData) || ~isnumeric(matrixData)
        error('Input must be a 2D numeric matrix.');
    end

    % Create the figure
    figure;

    % Draw the 3D bar chart
    h = bar3(matrixData);

    % Optional: set the color based on height
    for k = 1:length(h)
        zdata = get(h(k), 'ZData');
        cdata = zdata;
        set(h(k), 'CData', cdata, 'FaceColor', 'interp');
    end

    % Add labels and title
    xlabel('Column Index');
    ylabel('Row Index');
    zlabel('Value');
    title('3D Bar Plot of Matrix Values');

    % Improve view
    view(45, 30);
    colorbar;
end
