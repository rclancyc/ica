function [Zc, mu] = center_data(X);
    %   [Zc, mu] = center_data(X)
    %   arguments:  X (matrix to be whitened)
    %   returns:    Zc (centered data)
    %               mu (mean of each row)
    % This function will take time course data (each time step stored as a
    % column), and return the centered data for X.
    
    % find mean of rows
    mu = mean(X, 2); 
    
    % subtract row mean from each row.
    Zc = X - mu;
end