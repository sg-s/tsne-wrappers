function ydata = fit_p(P, labels, no_dims)
%fit_p Performs symmetric t-SNE on affinity matrix P
%
%   mappedX = fit_p(P, labels, no_dims)
%
% The function performs symmetric t-SNE on pairwise similarity matrix P 
% to create a low-dimensional map of no_dims dimensions (default = 2).
% The matrix P is assumed to be symmetric, sum up to 1, and have zeros
% on the diagonal.
% The labels of the data are not used by t-SNE itself, however, they 
% are used to color intermediate plots. Please provide an empty labels
% matrix [] if you don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
%
% (C) Laurens van der Maaten, 2010
% University of California, San Diego


if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Initialize some variables
n = size(P, 1);                                     % number of instances
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
max_iter = 1000;                                    % maximum number of iterations
epsilon = 500;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta

% Make sure P-vals are set properly
assert(~any(isnan(P(:))),'P matrix contains NaNs')
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
if ~initial_solution
    P = P * 4;                                      % lie about the P-vals to find better local minima
end

% use frozen random numbers
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1984)); 

% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end
y_incs  = zeros(size(ydata));
gains = ones(size(ydata));

tic; 

disp_iter = unique([ 1 2 3 round(linspace(5,max_iter,20))]);


% Run the iterations
for iter = 1:max_iter
    
    % Compute joint probability that point i and j are neighbors
    sum_ydata = sum(ydata .^ 2, 2);
    num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution

     % set diagonal to zero
    num(1:n+1:end) = 0;                                                

     % normalize to get probabilities
    Q = max(num ./ sum(num(:)), realmin);                              
    
    % Compute the gradients (faster implementation)
    L = (P - Q) .* num;
    y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
        
    % Update the solution
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
          + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    ydata = ydata + y_incs;
    ydata = bsxfun(@minus, ydata, mean(ydata, 1));
    
    % Update the momentum if necessary
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter && ~initial_solution
        P = P ./ 4;
    end

    cost = const - sum(P(:) .* log(Q(:)));

    
    % Print out progress
    if any(disp_iter == iter)
        t_elapsed = toc;
        t_total = (t_elapsed/iter)*max_iter;
        t_rem = t_total - t_elapsed;

        

        disp(['Iteration ' num2str(iter) ': error is ' num2str(cost) ' time remaining: ' num2str(t_rem)]);
    end
    
    % Display scatter plot (maximally first three dimensions)
    if ~rem(iter, 10) && ~isempty(labels)
        if no_dims == 1
            scatter(ydata, ydata, 9, labels, 'filled');
        elseif no_dims == 2
            scatter(ydata(:,1), ydata(:,2), 9, labels, 'filled');
        else
            scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 40, labels, 'filled');
        end
        axis tight
        axis off
        drawnow
    end
end
