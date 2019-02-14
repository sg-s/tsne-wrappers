classdef TSNE


properties

	implementation@TSNE.implementation = TSNE.implementation.internal
	distance_matrix@double
	raw_data@double
	num_dims = 2
	perplexity@double = 30

end



methods


	function R = fit(self)

		if self.implementation == TSNE.implementation.internal
			R = tsne(self.raw_data);

		elseif self.implementation == TSNE.implementation.vandermaaten
			if isempty(self.raw_data) && ~isempty(self.distance_matrix)
				% use distance matrix
				R = TSNE.vandermaaten.tsne_d(self.distance_matrix, [], self.num_dims, self.perplexity);
			else
				% use raw_data
			end
		else
			keyboard
		end


	end

end


end % classdef