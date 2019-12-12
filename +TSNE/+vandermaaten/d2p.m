function [P, beta] = d2p(D, perplexity, tol)
% D2P Identifies appropriate sigmas to get kk NNs up to some tolerance 
%
%   [P, beta] = d2p(D, kk, tol)
% 
% Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
% kernel with a certain uncertainty for every datapoint. The desired
% uncertainty can be specified through the perplexity u (default = 15). The
% desired perplexity is obtained up to some tolerance that can be specified
% by tol (default = 1e-4).
% The function returns the final Gaussian kernel in P, as well as the 
% employed precisions per instance in beta.
%
%
% (C) Laurens van der Maaten, 2008
% Maastricht University

    
disp(mfilename)

if ~exist('perplexity', 'var') || isempty(perplexity)
    perplexity = 15;
end
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-4; 
end

% Initialize some variables
n = size(D, 1);                     % number of instances
P = zeros(n);                    % empty probability matrix
P_off_diag = zeros(n-1,n);
beta = ones(n, 1);                  % empty precision vector
logU = log(perplexity);              % log of perplexity (= entropy)

% Run over all datapoints

if isempty(gcp('nocreate'))
    % no parallel pool
    for i = 1:n
        [P_off_diag(:,i), beta(i)] = TSNE.vandermaaten.d2p_parallel(logU, D(i,:), tol, i);
    end  
else

    parfor i = 1:n
        [P_off_diag(:,i), beta(i)] = TSNE.vandermaaten.d2p_parallel(logU, D(i,:), tol, i);
    end  

end

  

% rearrange into P
for i = 1:n
    P(i, [1:i - 1, i + 1:end]) = P_off_diag(:,i);
end



disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);



