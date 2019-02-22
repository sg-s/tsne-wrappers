function [thisP, thisBeta] = d2p_parallel(logU, Dslice, tol, i)

% remove the ith element from Dslice
Dslice(i) = [];

% Set minimum and maximum values for precision
betamin = -Inf; 
betamax = Inf;

% Compute the Gaussian kernel and entropy for the current precision
thisBeta = 1;
[H, thisP] = TSNE.vandermaaten.Hbeta(Dslice, thisBeta);

% Evaluate whether the perplexity is within tolerance
Hdiff = H - logU;
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
    Hdiff = H - logU;
    tries = tries + 1;
end
