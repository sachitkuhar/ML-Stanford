function [J, grad] = costFunction(theta,X,y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.



% Itialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

Z = X*theta ; 
g = 1./( 1 + exp(-Z));
for i=1:m
    J = J+ (-(y(i)*log(g(i)))-(1-y(i))*(log(1-g(i))))/m;
    grad = grad + ((g(i)-y(i))/m)*((X(i,:)'));
end


%t = zeros(size(X,1),1);
%v = zeros(size(X,1),1);
%q = zeros(size(X,1),size(theta,1));
%t = X*theta; 
%for n = 1:m
%v(n) = -y(n)*log(sigmoid(-t(n)))-(1-y(n))*log(sigmoid(-t(n))) ;

%end
%J = (1/m)*sum(v) ; 
%disp(J);
%for s=1:3
 %   for n=1:100
  %      q(n,s) = (1/m)*(sigmoid(-t(n,1))-y(n))*X(n,s);
   % end
%end

 %   grad = (sum(q))' ; 
%disp(size(sum(q)));
%disp(grad);

% =============================================================
end


