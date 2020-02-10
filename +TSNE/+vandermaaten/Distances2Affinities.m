function [P, const] = Distances2Affinities(D, perplexity, tol)

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
                    

% Make sure P-vals are set properly
assert(~any(isnan(P(:))),'P matrix contains NaNs')

P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
const = sum(P(:) .* log(P(:)));      