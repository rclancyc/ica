function [Z, T] = whiten_data(X)
    %   [Z, T] = whiten(X)
    %   arguments:  X (matrix to be whitened)
    %   returns:    Z (whitened data)
    %               T (transform matrix, i.e., Z = T*X)
    % This function will take time course data (each time step stored as a
    % column), and return the whitened data (decorrelated components with identity
    % covariance) as well as the whitening matrix, i.e., the matrix that
    % when applied to X gives a whitened r.v. Z
    
    % center data if not done so already
    if max(abs(mean(X, 2))) > 1e-10
        [X, ~] = center_data(X);
        fprintf('To track row mean of original data, center data before whitening. \n')
        fprintf('Centering data now... \n')
    end 
    
    [~, n] = size(X);
    [U, S, ~] = svd(X, 'econ');
    T = sqrt(n-1)*diag(1 ./ diag(S))*U';
    Z = T*X;
    
    % Note that we must scale by sqrt(n-1) to ensure identity covariance!!!
    % I wasted a lot of time on this.
end
