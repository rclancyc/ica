function [S, W, T, mu] = fastICA_in_progress(X, n_comps,  varargin) %nonlinearity, orthogonalization)
    % We will want to accept a number of optional arguments such as:
    %   max_iter: maximum # of iterations to run 
    %   print_every: print every so many iterations
    %   tol: tolerance for convergence
    %   nonlinearity: accept string for standard or user specified nonlinearity and gradient
    %   orthogonalization: deflationary, symmetric orthogonalization
    %
    prs = inputParser;
    addParameter(prs,'maxIts',1e3);
    addParameter(prs,'printEvery',1);
    addParameter(prs,'tol',1e-6);
    addParameter(prs,'func', []);
    addParameter(prs,'gradFunc', [])
    addParameter(prs,'orthogonalization', 'parallel')
    addParameter(prs,'nonlinearity', 'logcosh') 
        
    parse(prs,varargin{:});
    max_its = prs.Results.MaxIts;
    print_every = prs.Results.printEvery;
    tol = prs.Results.tol;
    g = prs.Results.func;
    g_p = prs.Results.gradFunc;
    orth = prs.Results.orthogonalization;
    nonlin = prs.Results.nonlinearity;

    if iscell(g) 
        % if function passed as a cell array, break into func and grad
        [g, g_p] = g{:};
    else 
        if ~isempty(g) && isempty(g_p)
            fprintf('Must provide gradient handle as well. \n')
        else
            % if no function handle passed, use prebuilt handles.
            if isempty(g)
                switch(nonlin)
                    case 'kurtosis'
                        g =@(x) 4*x.^3;
                        g_p =@(x) 12*x.^2;
                    case 'logcosh'
                        g =@(x) tanh(x);
                        g_p =@(x) 1-tanh(x).^2;
                    case 'exponential'
                        g =@(x) x.*exp(-x.^2/2);
                        g_p =@(x) (1-x.^2).*exp(-x.^2/2);              
                    otherwise
                        fprintf('Nonlinearity not recognized, choose "kurtosis", "logcosh", or "exponential". \n')
                end
            end
        end
    end
      
   [Z, mu] = center_data(X);
   [Z, T ] = whiten_data(Z);
    
    if strcmp(orth, 'parallel')
        % parallel orthogonalization
        [S, W] = symmetric_orthogonalization(g, g_p, Z, n_comps, tol, max_its);
    else
        % deflationary portion of code
        [S, W] = deflationary_orthogonalization(g, g_p, Z, n_comps, tol, max_its);
    end
    
    
end
    
  
function [S, W] = symmetric_orthogonalization(g, g_p, Z, n_comps, tol, max_it)
    [m, n_obs] = size(Z);
    it = 0;
    diff = inf;
    W = randn(n_comps, m);
    W = W ./ sqrt(sum(W.^2, 2));
    while diff > tol && it < max_it
        it = it + 1; 
        Wold = W;
        M = W*Z;

        % new iterate given by w_{k+1} = E[z*g(w_k'*z)] - E[g_p(w_k'*z)]*w_k.
        % Note we update each IC at once, each row gives a different IC. 
        W = g(M)*Z'/n_obs - diag(mean(g_p(M), 2))*W;  
        W = W ./ sqrt(sum(W.^2, 2));

        % Symmetric orthogonalization step
        % Note W = (WW')^{-1/2}W = (U*Sig^{-1}*U')(U*Sig*V') = U*V'
        [U, ~, V] = svd(W, 'econ');  
        W = U*V';                     

        % stopping criteria based on fact that if we have converged to IC's,
        % then the IC's won't change direction and therefore dot to +/-1.
        diff = max(1 - abs(dot(W, Wold,2)));
        fprintf('Diff = %4.6f \n', diff);
    end
    S = W*Z;
end


function [S, W] = deflationary_orthogonalization(g, g_p, Z, n_comps, tol, max_it)
    [m, n] = size(Z);
    W = randn(n_comps, m);
    W = W ./ sqrt(sum(W.^2, 2));
    % loop for each component
    for ic = 1:n_comps
        it = 0;
        diff = inf;
        w = randn(1,m);
        w = w/norm(w);
        while diff > tol && it < max_it
            wold = w;
            M = w*Z;
            w = g(M)*Z'/n - diag(mean(g_p(M), 2))*w;  % fixed point step
            
            if ic > 1
                % orthogonalize current iterate to previous IC's
                w = w - diag(w*W(1:(ic-1),:)')*W(1:(ic-1),:);
            end
            w = w/norm(w);
            
            diff = abs(1-abs(wold*w'));
        end
        W(ic, :) = w;
    end
    S = W*Z;
end   