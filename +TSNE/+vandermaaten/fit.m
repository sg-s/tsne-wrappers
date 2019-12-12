function [ydata,cost] = fit(X, params)
%TSNE Performs symmetric t-SNE on dataset X
%
%   mappedX = tsne(X, labels, no_dims, initial_dims, perplexity)
%   mappedX = tsne(X, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxD dataset X to reduce its 
% dimensionality to no_dims dimensions (default = 2). The data is 
% preprocessed using PCA, reducing the dimensionality to initial_dims 
% dimensions (default = 30). Alternatively, an initial solution obtained 
% from an other dimensionality reduction technique may be specified in 
% initial_solution. The perplexity of the Gaussian kernel that is employed 
% can be specified through perplexity (default = 30). The labels of the
% data are not used by t-SNE itself, however, they are used to color
% intermediate plots. Please provide an empty labels matrix [] if you
% don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego

disp(mfilename)


initial_dims = min(50, size(X, 2));

% Normalize input data
X = X - min(X(:));
X = X / max(X(:));
X = bsxfun(@minus, X, mean(X, 1));

% Perform preprocessing using PCA
if isempty(params.InitialSolution)
    disp('Preprocessing data using PCA...');
    if size(X, 2) < size(X, 1)
        C = X' * X;
    else
        C = (1 / size(X, 1)) * (X * X');
    end
    [M, lambda] = eig(C);
    [lambda, ind] = sort(diag(lambda), 'descend');
    M = M(:,ind(1:initial_dims));
    lambda = lambda(1:initial_dims);
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
    end
    X = bsxfun(@minus, X, mean(X, 1)) * M;
    clear M lambda ind
end

% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
% compute affinities using fixed perplexity
P = TSNE.vandermaaten.d2p(D, params.perplexity, 1e-5);                                           
clear D

% Run t-SNE
[ydata,cost] = TSNE.vandermaaten.fit_p(P, params);
