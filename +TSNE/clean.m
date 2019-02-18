% removes NaNs and Inf from the data martix

function X = clean(X)

% rotate correctily
sz = size(X);
if sz(1) > sz(2)
	X = X';
end

X(:,isnan(sum(X))) = [];
X(:,isinf(sum(X))) = [];