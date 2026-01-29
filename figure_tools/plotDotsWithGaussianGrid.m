function plotDotsWithGaussianGrid(matrixData)
% PLOTDOTSWITHGAUSSIANGRID - Draws 3D dots representing a matrix
% and overlays a fitted 2D Gaussian grid surface.
%
% Input:
%   matrixData - A 2D numeric matrix with positive values.

    if ~ismatrix(matrixData) || ~isnumeric(matrixData)
        error('Input must be a 2D numeric matrix.');
    end

    % Transpose for visual alignment (as in bar3)
    Z = matrixData';  
    [rows, cols] = size(Z);
    [X, Y] = meshgrid(1:cols, 1:rows);

    % Flatten for fitting
    xData = X(:);
    yData = Y(:);
    zData = Z(:);

    % Fit 2D Gaussian
    gauss2D = @(p, x, y) ...
        p(1) * exp(-(((x - p(2)).^2) / (2 * p(4)^2) + ((y - p(3)).^2) / (2 * p(5)^2))) + p(6);
    initParams = [max(zData), mean(xData), mean(yData), std(xData), std(yData), min(zData)];
    objFun = @(p) sum((gauss2D(p, xData, yData) - zData).^2);
    fittedParams = fminsearch(objFun, initParams, optimset('Display', 'off'));
    Zfit = gauss2D(fittedParams, X, Y);

    % Plot 3D dots
    figure;
    scatter3(X(:), Y(:), Z(:), 50, Z(:), 'filled'); % size=50, color by height
    hold on;

    % Overlay Gaussian wireframe
    mesh(X, Y, Zfit, ...
         'EdgeColor', 'k', ...
         'FaceColor', 'none', ...
         'LineStyle', '-', ...
         'LineWidth', 1.0);

    % Style
    xlabel('Column Index');
    ylabel('Row Index');
    zlabel('Value');
    title('3D Dots with Fitted Gaussian Wireframe');
    colormap turbo;
    colorbar;
    view(45, 30);
    grid on;
end
