function C = confusion_matrix(labels, predictions, numClasses)
% Creates a confusion matrix given a set of labels and predictions.
%
% C(i,j) = is the percentage of images from class i that
% were classified as class j.
%
% C = confusionMatrix(truecategoruy, votedCategory);
%
% Inputs:
%   labels - a vector of discrete integer labels. The smallest label should be
%            no smaller than 1.
%   predictions - a vector of discrete integer predictions. The smallest
%                 prediction should be no smaller than 1.
%
% Ouputs:
%   C - the confusion matrix.

error(nargchk(2, 3, nargin));

if numel(labels) ~= numel(predictions)
  error('Oops. The number of labels and predictions must be equal.');
end

% Flatten both in case one is a row and the other is a column vector.
labels = labels(:);
predictions = predictions(:);

if nargin == 3
  n = numClasses;
  m = numClasses;
else
  n = max(labels);
  m = max(predictions);
end
  
C = zeros([n m]);

for i = 1:n
  for j = 1:m
    if sum(labels == i) > 0
      C(i,j) = sum((labels == i) .* (predictions == j));
    end
  end
end

