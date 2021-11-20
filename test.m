% ICA test
rng(1);
T = 10000;
ts = linspace(0,2*pi, T);

S = zeros(2,T);
S(1,:) = sign(randn(1,T)).*exprnd(1,1,T);
S(2,:) = 8*(rand(1,T)-0.5);

%s(1,:) = sin(10*pi*ts).^2;
%S(2,:) = sin(3*ts+.001);



A = rand(2);
X = A*S;

[Xc,mu] = center_data(X);
[Z,T] = whiten_data(Xc);


% We have observed whitened signals. Now we want to search the space of
% vectors such that when we project z onto it, we have a random variable
% that is as non-gaussian as possible.
figure(8)
thetas = linspace(0,2*pi,100);
bins = linspace(-6,6,100);
kurt = zeros(length(thetas));
min_kurt = inf;
max_kurt = -inf;
for i = 1:100
    th = thetas(i);
    w = [cos(th); sin(th)];
    Y = w'*Z;
    
    kurt(i) = mean(Y.^4) - 3*(mean(Y.^2))^2;
    if kurt(i) > max_kurt
        max_kurt = kurt(i);
        i_kurt_max = i;
        wmax = w;
    end
    if kurt(i) < min_kurt
        min_kurt = kurt(i);
        i_kurt_min = i;
        wmin = w;
    end
    
    hist(Y, bins)
    xlim([-6,6])
    ylim([0,1000])
    xlabel("y = w'*z")
    ylabel('Frequency')
    title(strcat('KURTOSIS = ', num2str(round(kurt(i),4))))
    pause(.01)
end

figure(9);
subplot(211);
hist(wmin'*Z, bins)
xlim([-6,6])
ylim([0,1000])
xlabel("y = w'*z")
ylabel('Frequency')
title(strcat('MIN KURTOSIS = ', num2str(round(min_kurt,4))))

subplot(212);
hist(wmax'*Z, bins)
xlim([-6,6])
ylim([0,1000])
xlabel("y = w'*z")
ylabel('Frequency')
title(strcat('MAX KURTOSIS = ', num2str(round(max_kurt,4))))

% we found projections that give minimum and maximum kurtosis. Using these
% as the rows of the unmixing matrix, we can look at the unmixed matrix. 
%%
figure(10)
subplot(221)
scatter(S(1,:), S(2,:))
title('Scatter plot of v_1 vs v_2')

subplot(222)
scatter(X(1,:), X(2,:))
title('Signal mixture')

W = [wmin';wmax'];
Xest = W*Z;
subplot(223)
scatter(Xest(1,:), Xest(2,:))
title('Unmixed with ICA')

[V,D] = eig(Z*Z');

subplot(224)
Xest=V*Z;
scatter(Xest(1,:), Xest(2,:))
title('"Unmixed" with PCA')

eta = 1e-3;
% converged = false;
% while ~converged
%     y = w'*Z;
%     kur = mean(y.^4) - 3*mean(y.^2))^2;
%     %w = w - eta * 4 * sign(kur)*( 
% end
















