
% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H, P] = Hbeta(D, beta)
	
P = exp(-D * beta);
sumP = sum(P);
H = log(sumP) + beta * sum(D .* P) / sumP;
P = P / sumP;