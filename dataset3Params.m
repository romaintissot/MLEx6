function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

testValues = [0.01 0.03 0.1 0.3 1 3 10 30 100 300];
CVerrormin = inf;
CVerror = inf;
Cmin = 0;
sigmamin = 0;
essai = size(testValues).^2;


for C=testValues
	for sigma=testValues
		fprintf('Test C = %f\n',C);
		fprintf('Test sigma = %f\n',sigma);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions=svmPredict (model, Xval);
		CVerror = mean(double(predictions ~= yval));
		if CVerror <= CVerrormin
			CVerrormin = CVerror;
			
			Cmin = C;
			sigmamin = sigma;
			fprintf('New minimum error = %f, C = %f and sigma = %f.\n',CVerror, Cmin, sigmamin);
		end
		
	end
end

C=Cmin;
sigma = sigmamin;

fprintf('Best value for C = %f and sigma = %f', C, sigma);




% =========================================================================

end
