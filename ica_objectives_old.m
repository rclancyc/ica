
% set # of compenents, # of meas. in time, beginning and end times
n_sources = 2;
n_mixtures = 10;
n_T = 1001;
T_i = 0;
T_f = 1;

% create times vector and mixing matrix
t = linspace(T_i, T_f, n_T);
A = randn(n_mixtures, n_sources);
Anorms = sqrt(sum(A.^2, 2));
A = bsxfun(@rdivide, A, Anorms);

% create TRUE signal matrix 
S = zeros(n_sources, n_T);
amp = 9*rand(n_sources,1) + 1;
freq = 10*rand(n_sources,1)+1;
phase = 2*randn(n_sources,1);
for i = 1:n_sources
    S(i,:) = amp(i)*sin(2*pi*freq(i)*t + phase(i));
end
S(1,:) = sign(1*sin(2*pi*freq(1)*t + phase(1)));
S(2,:) = 1*sawtooth(2*pi*freq(2)*t + phase(2));
%S(3,:) = 1*sin(2*pi*freq(3)*t + phase(3));
S = S +.05*randn(size(S));

S = S - mean(S,2);

% create signal mixture
X = A*S;

Xrow_mean = mean(X,2);

% whiten mixture
[Z, T] = whiten_data(X);

% now we have new data Z = T*X = T*A*S = M*S. Goal is to find M and S.
% Let M^{-1} = W
M = T*A;
%W_true = inv(M); 
%Y_true = W_true*Z; % want to know kurtosis of each row of Y (for fixed w)
Y_true = M \ Z; % if we have problem, use code in line above. 


% now estimate kurtosis and see how we do. W_est is appx. M^{-1}
[W_est, S_est] = fastICA(Z, n_sources, 'logcosh', 'parallel');

kurt_true = zeros(n_sources, 1);
kurt_est = zeros(n_sources, 1);

for i = 1:n_sources
    y_true = Y_true(i, :);
    kurt_true(i) = mean(y_true.^4) - 3*(mean(y_true.^2))^2;
    
    y_est = S_est(i, :);
    kurt_est(i) = mean(y_est.^4) - 3*(mean(y_est.^2))^2;
end






% what is the estimated kurtosis using true mixing matrix and true S.
% kurt(w'*z) = E((w'*z)^4) - 3 (E(w'*z)^2)^2


figure(1);
for i = 1:n_sources
    subplot(n_sources, 2, 2*(i-1)+1);
    plot(t, S(i,:));
end

for i = 1:n_sources
    subplot(n_sources, 2, 2*i);
    plot(t, W_est(i,:)*Z);
end

    
%norm(W_est - inv(A'*A)*A')


kt = kurt_true/norm(kurt_true);
ke = kurt_est/norm(kurt_est);
[sort(abs(kt)), sort(abs(ke))]


