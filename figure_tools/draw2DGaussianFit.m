function draw2DGaussianFit(matrixData)
% DRAW2DGAUSSIANFIT - Fits and draws a transparent 2D Gaussian surface over a matrix
%
% Syntax: draw2DGaussianFit(matrixData)
%
% Input:
%   matrixData - A 2D numeric matrix representing surface values

    % Validate input
    if ~ismatrix(matrixData) || ~isnumeric(matrixData)
        error('Input must be a 2D numeric matrix.');
    end

    % Get size of matrix
    [rows, cols] = size(matrixData);
    [X, Y] = meshgrid(1:cols, 1:rows);
    Z = double(matrixData);

    % Flatten for fitting
    xData = X(:);
    yData = Y(:);
    zData = Z(:);

    % Initial parameters: amplitude, center_x, center_y, sigma_x, sigma_y, offset
    amp = max(zData);
    cx = mean(xData);
    cy = mean(yData);
    sx = std(xData);
    sy = std(yData);
    offset = min(zData);
    
    % Gaussian model function
    gauss2D = @(params, x, y) ...
        params(1) * exp(-(((x - params(2)).^2) / (2 * params(4)^2) + ((y - params(3)).^2) / (2 * params(5)^2))) + params(6);

    % Objective function
    objFun = @(params) sum((gauss2D(params, xData, yData) - zData).^2);

    % Optimization
    initialGuess = [amp, cx, cy, sx, sy, offset];
    options = optimset('Display','off');
    fittedParams = fminsearch(objFun, initialGuess, options);

    % Evaluate fitted Gaussian
    Zfit = reshape(gauss2D(fittedParams, X(:), Y(:)), size(X));

    % Plot
    figure;
    surf(X, Y, Zfit, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    hold on;
    mesh(X, Y, Z, 'EdgeColor', [0.3 0.3 0.3]); % Optional: show original data mesh
    colormap jet;
    colorbar;
    view(45, 30);
    xlabel('Column Index');
    ylabel('Row Index');
    zlabel('Value');
    title('Fitted 2D Gaussian (Transparent) and Original Surface');
end
