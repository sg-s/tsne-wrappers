classdef TSNE < Hashable


properties

	implementation@TSNE.implementation = TSNE.implementation.internal
	distance_matrix@double
	raw_data@double
	num_dims = 2
	perplexity@double = 30

	% cache embeddings to immediately recall it next time?
	use_cache@logical = true

	n_cores@double = mtools.core.numcores;

	n_iter@double = 1e3;

end



methods


	function R = fit(self)

		% first check the hash
		H = self.hash;
		cache_dir = which(mfilename);
		cache_dir = [fileparts(cache_dir) filesep '.cache'];
		if ~isdir(cache_dir)
			mkdir(cache_dir)
		end

		if exist([cache_dir filesep H '.cache'],'file') == 2
			load([cache_dir filesep H '.cache'],'R','-mat');
			return
		end


		if self.implementation == TSNE.implementation.internal
			R = tsne(self.raw_data);

		elseif self.implementation == TSNE.implementation.vandermaaten
			if isempty(self.raw_data) && ~isempty(self.distance_matrix)
				% use distance matrix
				R = TSNE.vandermaaten.tsne_d(self.distance_matrix, [], self.num_dims, self.perplexity);
			else
				% use raw_data
				keyboard
			end
		elseif self.implementation == TSNE.implementation.multicore
			R = TSNE.multicore.mctsne(self.raw_data);

		else
			keyboard
		end

		% cache if need be
		if self.use_cache
			save([cache_dir filesep H '.cache'],'R','-nocompression')
		end


	end


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