function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)

m = length(y);
J_history = zeros(iterations, 1);

for iter = 1:iterations
    delta = X*theta - y;
    theta(1) = theta(1) - (alpha/m)*sum(delta);
    tmp = 0;
    for jter = 1:m
        tmp = tmp + delta(jter)*X(jter, 2);
    end

    theta(2) = theta(2) - (alpha/m)*tmp;

    J_history(iter) = computeCost(X, y, theta);

end

end
