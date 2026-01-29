function Q = quantize_matrix_values(M, intervals)
% QUANTIZE_MATRIX_VALUES  Quantize the values in M without reducing dimensions.
%
% For each element x in M:
%   if 5.91 <= x <= 6.9, then x becomes 6,
%   if 4.91 <= x <= 5.9, then x becomes 5,
%   if 3.91 <= x <= 4.9, then x becomes 4,
%   if 2.91 <= x <= 3.9, then x becomes 3,
%   if 1.91 <= x <= 2.9, then x becomes 2,
%   if 0.91 <= x <= 1.9, then x becomes 1,
%   otherwise, x remains unchanged.
%
% Input:
%   M  - an m×n matrix of distances or cost values.
%
% Output:
%   Q  - an m×n matrix with quantized values.

    Q = M;  % Copy the input to the output
    
    % Define the quantization intervals and corresponding values.
    % Each row: [lower_bound, upper_bound, new_value]
    % intervals = [0, 6.9, 1];


    % Loop through each interval
    for i = 1:size(intervals, 1)
        lower_bound = intervals(i, 1);
        upper_bound = intervals(i, 2);
        new_val = intervals(i, 3);
        
        % Create a logical mask for values within the current interval.
        mask = (M >= lower_bound) & (M <= upper_bound);
        Q(mask) = new_val;
    end

    % % Display the quantized output matrix
    % disp('Quantized Output Matrix:');
    % disp(Q);
end





% function Q = quantize_matrix_values(M)
% % QUANTIZE_MATRIX_VALUES  Quantize the values in M without reducing dimensions.
% %
% % For each element x in M:
% %   if 5.91 <= x <= 6.9, then x becomes 6,
% %   if 4.91 <= x <= 5.9, then x becomes 5,
% %   if 3.91 <= x <= 4.9, then x becomes 4,
% %   if 2.91 <= x <= 3.9, then x becomes 3,
% %   if 1.91 <= x <= 2.9, then x becomes 2,
% %   if 0.91 <= x <= 1.9, then x becomes 1,
% %   otherwise, x remains unchanged.
% %
% % Input:
% %   M  - an m×n matrix of distances or cost values.
% %
% % Output:
% %   Q  - an m×n matrix with quantized values.
% 
%     Q = M;  % Copy the input to the output
% 
%     % Define the quantization intervals and corresponding values.
%     % Each row: [lower_bound, upper_bound, new_value]
%     % intervals = [0, 6.9, 1];
% 
% 
%     intervals = [5.91, 6.9, 6;
%                  4.91, 5.9, 5;
%                  3.91, 4.9, 4;
%                  2.91, 3.9, 3;
%                  1.91, 2.9, 2;
%                  0.91, 1.9, 1];
% 
%     % 
%     % intervals = [6.5, 6.9, 6.5;
%     %              5.91, 6.49, 6;
%     %              5.5, 5.9, 5.5;
%     %              4.91, 5.49, 5;
%     %              4.5, 4.9, 4.5;
%     %              3.91, 4.49, 4;
%     %              3.5, 3.9, 3.5;
%     %              2.91, 3.49, 3;
%     %              2.5, 2.9, 2.5;
%     %              1.91, 2.49, 2;
%     %              1.5, 1.9, 1.5;
%     %              0.91, 1.49, 1];
% 
% 
%     % Loop through each interval
%     for i = 1:size(intervals, 1)
%         lower_bound = intervals(i, 1);
%         upper_bound = intervals(i, 2);
%         new_val = intervals(i, 3);
% 
%         % Create a logical mask for values within the current interval.
%         mask = (M >= lower_bound) & (M <= upper_bound);
%         Q(mask) = new_val;
%     end
% 
%     % % Display the quantized output matrix
%     % disp('Quantized Output Matrix:');
%     % disp(Q);
% end
