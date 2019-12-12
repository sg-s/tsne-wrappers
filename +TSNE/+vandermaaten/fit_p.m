function [ydata, cost] = fit_p(P, self)
%fit_p Performs symmetric t-SNE on affinity matrix P
%
%   mappedX = fit_p(P, labels, self.NumDims)
%
% The function performs symmetric t-SNE on pairwise similarity matrix P 
% to create a low-dimensional map of self.NumDims dimensions (default = 2).
% The matrix P is assumed to be symmetric, sum up to 1, and have zeros
% on the diagonal.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego



% unpack
Alpha = self.Alpha;
min_gain = self.MinGain;
momentum = self.InitialMomentum;
epsilon = self.Epsilon;

% Initialize some variables
n = size(P, 1);                                     % number of instances
                 
                                    

% Make sure P-vals are set properly
assert(~any(isnan(P(:))),'P matrix contains NaNs')

P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
const = sum(P(:) .* log(P(:)));                     % constant in KL divergence


if isempty(self.InitialSolution)
    P = P * 4;                                      % lie about the P-vals to find better local minima
end

% use frozen random numbers
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',self.RandomSeed)); 

% Initialize the solution
if isempty(self.InitialSolution)
    ydata = .0001 * randn(n, self.NumDims);
end
y_incs  = zeros(size(ydata));
gains = ones(size(ydata));

tic; 

disp_iter = unique([ 1 2 3 round(linspace(5,self.NIter,20))]);


% Run the iterations
for iter = 1:self.NIter
    
    % Compute joint probability that point i and j are neighbors

    % This commented out code is what van der Maaten wrote. Not clear why
    % this is written in so convoluted a fashion
    % tic
    % sum_ydata = sum(ydata .^ 2, 2);
    % num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
    % % set diagonal to zero
    % num(1:n+1:end) = 0;  
    % toc



    d = pdist(ydata);
    if Alpha == 1
        % normal Cauchy distribution (Student's t-distribution with Alpha = 1)
        num = squareform(1./(1+d.^2));
    else
        % generalized t-Distribution
        num = squareform((1 + ((d.^2)/Alpha)).^(-Alpha));
    end


                                                   

     % normalize to get probabilities
    Q = max(num ./ sum(num(:)), realmin);                              
    
    % Compute the gradients (faster implementation)
    L = (P - Q) .* (num.^(1/Alpha));
    y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
        
    % Update the solution
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
          + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    ydata = ydata + y_incs;
    ydata = bsxfun(@minus, ydata, mean(ydata, 1));
    
    % Update the momentum if necessary
    if iter == self.MomSwitchIter
        momentum = self.FinalMomentum;
    end
    if iter == self.StopLyingIter && isempty(self.InitialSolution)
        P = P ./ 4;
    end

    cost = const - sum(P(:) .* log(Q(:)));

    
    % Print out progress
    if any(disp_iter == iter)
        t_elapsed = toc;
        t_total = (t_elapsed/iter)*self.NIter;
        t_rem = t_total - t_elapsed;

        disp(['Iteration ' num2str(iter) ': error is ' num2str(cost) ' time remaining: ' num2str(t_rem)]);
    end
    
end
