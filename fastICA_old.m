function [W, S] = fastICA(Z, n_comps, nonlinearity, orthogonalization)
    [n_mixtures, n] = size(Z);
    if Z(1,:)'*Z(2,:) > 1e-12
        Z = whiten(Z);
    end
    
    % determine non-linearity to use. 
    if strcmp(nonlinearity, 'kurtosis')
        G =@(x) 0.25.*x^4;
        g =@(x) x.^3;
        g_p =@(x) 3.*x.^2;
    elseif strcmp(nonlinearity, 'logcosh')
        G =@(x) log(cosh(x)); 
        g =@(x) tanh(x);
        g_p =@(x) 1-tanh(x).^2
    elseif strcmp(nonlinearity, 'exponential')
        G =@(x) -exp(-x.^2/2);
        g =@(x) x.*exp(-x.^2/2);
        g_p =@(x) (1-x.^2).*exp(-x.^2/2); 
    end 
    
    
    S = zeros(size(Z));
    max_its = 1e4;
    tol = 1e-6;

    if strcmp(orthogonalization, 'deflationary')
        W0 = randn(n_comps, n_mixtures);
        W = get_IC(Z, W0, n_comps, g, g_p, tol, max_its);
        S = W*Z;
    else
        
        % if not deflationary or mention, just use symmetric orthogonalization

        % randomly initialize with orthonormal columns
        %W = randn(m, n_comps);
        %[W, ~] = qr(W);
        W = eye(n_mixtures);
        W = W(:,1:n_comps);
        
        W = W';
        diff2 = inf;
        it_count2 = 0;
        %W*W'
        while diff2 > tol && it_count2 < max_its
            it_count2 = it_count2 + 1;
            %W = get_IC(Z, W, n_comps, g, g_p, tol, max_its)
            for k = 1:n_comps
                diff = inf;
                it_count = 0;    
                w = W(k, :)';
                % this can be parallelized and should be.
                while diff > tol && it_count < max_its          
                    w_old = w;
                    it_count = it_count + 1;
                    y = w'*Z;
                    w = mean(Z.*repmat(g(y),n_mixtures,1),2) - mean(g_p(y))*w;
                    w = w/norm(w);
                    diff = abs(1 - abs(w_old'*w));
                end
                W(k, :) = w;
            end

            [E, Sig, ~] = svd(W);
            A = E*diag(1 ./ diag(Sig))*E';
            %W*W'
            W_old = W;
            W = A*W;

            %W = W/norm(W,1);
            %W = 1.5*W - W*(W*W')*W;

            diff2 = norm(W*W_old'-eye(n_comps), 1);
            fprintf('Error is %4.6f \n', diff2);
        
        end
        S = W*Z;
    end
end
  


function W = get_IC(Z, W, n_comps, func, grad, tol, max_its)
    [m,~] = size(Z);
    [n_comps, ~] = size(W);
    g = func;
    g_p = grad;
   
    for k = 1:n_comps
        w = W(k,:)';
        w = w/norm(w);
        diff = inf;
        it_count = 0;
        while diff > tol && it_count < max_its 
            it_count = it_count + 1;

            w_old = w;
            % for y = w*z, calculate E(y^4)
            y = w'*Z;
            w = mean(Z.*repmat(g(y),m,1),2) - mean(g_p(y))*w;
            w = w/norm(w);

            if k > 1
                for j = 1:(k-1)
                    w = w - (W(j, :)*w)*W(j, :)';
                end
            end
            w = w/norm(w);

            diff = abs(1 - abs(w_old'*w));
        end
        if diff > tol 
            fprintf('Failed to convere on component %i, consider increasing iteration or increasing tolerance \n', k);
        end
        W(k, :) = w;
     
    end
end






    