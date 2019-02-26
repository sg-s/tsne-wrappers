classdef TSNE < Hashable


properties

	implementation@TSNE.implementation = TSNE.implementation.internal
	distance_matrix@double
	raw_data@double
	num_dims = 2
	perplexity@double = 30

	% cache embeddings to immediately recall it next time?
	use_cache@logical = true

	n_cores@double = corelib.numcores;

	n_iter@double = 1e3;

end



methods





	function self = set.raw_data(self, value)

		% should be a matrix
		assert(ismatrix(value),'raw_data should be a matrix')

		% rotate correctily
		sz = size(value);
		if sz(1) > sz(2)
			value = value';
		end

		% check for NaNs and Infs
		assert(~any(isnan(sum(value))),'NaNs found in raw data. Use TSNE.clean() to clean it before setting')

		assert(~any(isinf(sum(value))),'Inf found in raw data. Use TSNE.clean() to clean it before setting')

		self.raw_data = value;

	end

end


end % classdef