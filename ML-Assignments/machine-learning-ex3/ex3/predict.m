function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% fprintf('\nX: %f\n', X(1:10));
% fprintf('\nXEnd: %f\n', X(end-10: end));
% fprintf('\nTheta1: %f\n', Theta1);
% fprintf('\nTheta1End: %f\n', Theta1(end-10: end));
% fprintf('\nTheta2: %f\n', Theta2(1:10));
% fprintf('\nTheta2End: %f\n', Theta2(end-10:end));

temp1 = [ones(m, 1) X];
temp2 = [ones(m, 1) sigmoid(temp1 * Theta1')];
% fprintf('\nones: %f\n', (sigmoid(temp1 * Theta1'))(1:10));
% fprintf('\no sig: %f\n', (temp1 * Theta1')(1:10));
% fprintf('\nfirst: ', disp((temp1 * Theta1')(1:10)));
% disp("The value of pi is:"), disp((temp1 * Theta1')(1:10))
% fprintf('\nones: %f\n', (temp1 * Theta1')(end - 20:end));
temp3 = sigmoid(temp2 * Theta2');
[maxTemp3, maxTemp3_2] = max(temp3');
p = maxTemp3_2';
fprintf('p: %f\n', (temp3)(end-10:end));
% fprintf('pend: %f\n', p(end-10:end));





% =========================================================================


end
