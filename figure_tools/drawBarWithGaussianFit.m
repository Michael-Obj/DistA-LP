function drawBarWithGaussianFit(matrixData)
% PLOTBARWITHGAUSSIANGRID - Draws a 3D bar plot with an overlaid 2D Gaussian grid surface
%
% Input:
%   matrixData - 2D numeric matrix

    if ~ismatrix(matrixData) || ~isnumeric(matrixData)
        error('Input must be a 2D numeric matrix.');
    end

    % Transpose the matrix to match bar3's display orientation
    Z = matrixData';  % Transpose to align with bar3 orientation
    [rows, cols] = size(Z);
    [X, Y] = meshgrid(1:cols, 1:rows);

    % Flatten data for fitting
    xData = X(:);
    yData = Y(:);
    zData = Z(:);

    % Define 2D Gaussian
    gauss2D = @(p, x, y) ...
        p(1) * exp(-(((x - p(2)).^2) / (2 * p(4)^2) + ((y - p(3)).^2) / (2 * p(5)^2))) + p(6);

    % Initial guess
    p0 = [max(zData), mean(xData), mean(yData), std(xData), std(yData), min(zData)];

    % Fit
    objFun = @(p) sum((gauss2D(p, xData, yData) - zData).^2);
    fittedParams = fminsearch(objFun, p0, optimset('Display', 'off'));

    % Compute fitted Gaussian
    Zfit = gauss2D(fittedParams, X, Y);

    % Plot 3D bar
    figure;
    h = bar3(matrixData);
    hold on;

    for k = 1:length(h)
        zdata = get(h(k), 'ZData');
        set(h(k), 'CData', zdata, 'FaceColor', 'interp');
    end

    % Overlay Gaussian wireframe using mesh
    mesh(X, Y, Zfit, ...
         'EdgeColor', 'k', ...
         'FaceColor', 'none', ...
         'LineStyle', '-', ...
         'LineWidth', 1.0);

    % Style
    xlabel('Column Index');
    ylabel('Row Index');
    zlabel('Value');
    title('3D Bar Plot with Fitted Gaussian Wireframe');
    colormap turbo;
    colorbar;
    view(45, 30);
    grid on;
end
