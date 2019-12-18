
function [ydata, cost] = fit_d(D, params)
%fit_d Performs symmetric t-SNE on the pairwise Euclidean distance matrix D
%
%   mappedX = fit_d(D, labels, no_dims, perplexity)
%   mappedX = fit_d(D, labels, initial_solution, perplexity)
%
% The function performs symmetric t-SNE on the NxN pairwise Euclidean 
% distance matrix D to construct an embedding with no_dims dimensions 
% (default = 2). An initial solution obtained from an other dimensionality 
% reduction technique may be specified in initial_solution. 
% The perplexity of the Gaussian kernel that is employed can be specified 
% through perplexity (default = 30). The labels of the data are not used 
% by t-SNE itself, however, they are used to color intermediate plots. 
% Please provide an empty labels matrix [] if you don't want to plot 
% results during the optimization.
% The data embedding is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego



% normalize distances
D = D / max(D(:));  

% Compute joint probabilities     
% compute affinities using fixed perplexity    
P = TSNE.vandermaaten.d2p(D .^ 2, params.perplexity, params.Tolerance, params.UseParallel);      

% Run t-SNE
[ydata, cost] = TSNE.vandermaaten.fit_p(P, params);

