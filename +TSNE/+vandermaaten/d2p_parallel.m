function [thisP, thisBeta] = d2p_parallel(desired_entropy, Dslice, tol, i)

% remove the ith element from Dslice
% we're doing this because this is the easisest way to parallelize this
% operation (and to avoid D being a broadcast variable)
Dslice(i) = [];

% Set minimum and maximum values for precision
betamin = -Inf; 
betamax = Inf;

% Compute the Gaussian kernel and entropy for the current precision
thisBeta = 1;
[H, thisP] = TSNE.vandermaaten.Hbeta(Dslice, thisBeta);

% Evaluate whether the perplexity is within tolerance
Hdiff = H - desired_entropy;
tries = 0;
while abs(Hdiff) > tol && tries < 50

    % If not, increase or decrease precision
    if Hdiff > 0
        betamin = thisBeta;
        if isinf(betamax)
            thisBeta = thisBeta * 2;
        else
            thisBeta = (thisBeta + betamax) / 2;
        end
    else
        betamax = thisBeta;
        if isinf(betamin) 
            thisBeta = thisBeta / 2;
        else
            thisBeta = (thisBeta + betamin) / 2;
        end
    end
    
    % Recompute the values
    [H, thisP] = TSNE.vandermaaten.Hbeta(Dslice, thisBeta);
    Hdiff = H - desired_entropy;
    tries = tries + 1;
end

% if thisP has NaNs, then perplexity is probably too low
assert(~any(isnan(thisP)),'NaNs appear in P. This probably means that perplexity is too low')