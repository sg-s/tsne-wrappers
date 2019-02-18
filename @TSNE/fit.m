function R = fit(self)

% first check the hash
H = self.hash;
cache_dir = fileparts(fileparts(which('TSNE.implementation')));
cache_dir = [cache_dir filesep '.cache'];
if ~isdir(cache_dir)
	mkdir(cache_dir)
end

if exist([cache_dir filesep H '.cache'],'file') == 2 && self.use_cache
	load([cache_dir filesep H '.cache'],'R','-mat');
	return
end


if self.implementation == TSNE.implementation.internal
	R = tsne(self.raw_data');

elseif self.implementation == TSNE.implementation.vandermaaten
	if isempty(self.raw_data) && ~isempty(self.distance_matrix)
		% use distance matrix
		R = TSNE.vandermaaten.fit_d(self.distance_matrix, [], self.num_dims, self.perplexity);
	else
		% use raw_data
		R = TSNE.vandermaaten.fit(self.raw_data', [], self.num_dims, [], self.perplexity);

	end
elseif self.implementation == TSNE.implementation.multicore
	R = TSNE.multicore.fit(self.raw_data)';
elseif self.implementation == TSNE.implementation.berman

	R = TSNE.berman.fit(self.raw_data','perplexity',self.perplexity);

else
	keyboard
end

% cache if need be
if self.use_cache
	save([cache_dir filesep H '.cache'],'R','-nocompression')
end


