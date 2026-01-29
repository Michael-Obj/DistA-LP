% function [F1, F2, F3] = generate_gaussians(n, m, params)
%     [I1, J1] = meshgrid(linspace(0,1,n));
%     [I2, J2] = meshgrid(1:m, 1:n);
% 
% 
%     F1 = gaussian2d(I1, J1, params(1), params(2), params(3), params(4));
%     F2 = gaussian2d(I2, J2, params(5), params(6), params(7), params(8));
%     F3 = gaussian2d(I2, J2, params(9), params(10), params(11), params(12));
% end
% 
% 
% function val = gaussian2d(x, y, cx, cy, sx, sy)
%     val = exp(-((x - cx).^2 ./ (2 * sx^2) + (y - cy).^2 ./ (2 * sy^2)));
% end


function [G1,G2,G3] = generate_gaussians(n, m, theta)
% Same parameter order you used before, but on a [0,1]×[0,1] grid.

% grids -----------------------------------------------------------
[y1,x1] = ndgrid(linspace(0,1,n));                      % n × n
[y2,x2] = ndgrid(linspace(0,1,n), linspace(0,1,m));     % n × m

% unpack ----------------------------------------------------------
c1x = theta(1);  c1y = theta(2);  s1x = theta(3);  s1y = theta(4);
c2x = theta(5);  c2y = theta(6);  s2x = theta(7);  s2y = theta(8);
c3x = theta(9);  c3y = theta(10); s3x = theta(11); s3y = theta(12);

% three diagonal-covariance Gaussians -----------------------------
G1 = single_gauss(x1,y1,c1x,c1y,s1x,s1y);
G2 = single_gauss(x2,y2,c2x,c2y,s2x,s2y);
G3 = single_gauss(x2,y2,c3x,c3y,s3x,s3y);
end
% ----------------------------------------------------------------------
function G = single_gauss(x,y,cx,cy,sx,sy)
dx = (x - cx) ./ sx;
dy = (y - cy) ./ sy;
G  = exp(-0.5 * (dx.^2 + dy.^2));
end