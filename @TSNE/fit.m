% embed data or distance matrix
% 
function [R, cost] = fit(self)

% first check the hash
if self.UseCache
	H = self.hash;
	cache_dir = fileparts(fileparts(which('TSNE.implementation')));
	cache_dir = [cache_dir filesep '.cache'];
	if ~isdir(cache_dir)
		mkdir(cache_dir)
	end

	if exist([cache_dir filesep H '.cache'],'file') == 2 && self.UseCache
		load([cache_dir filesep H '.cache'],'R','-mat');
		load([cache_dir filesep H '.cache'],'cost','-mat');
		if ~exist('cost','var')
			cost = NaN;
		end
		return
	end
end



if self.implementation == TSNE.implementation.internal
	[R, cost] = tsne(self.RawData');

elseif self.implementation == TSNE.implementation.vandermaaten
	if isempty(self.RawData) && ~isempty(self.DistanceMatrix)
		% use distance matrix

		if isempty(self.InitialSolution)
			[R, cost] = TSNE.vandermaaten.fit_d(self.DistanceMatrix,self.getParameters);
		else
			[R, cost] = TSNE.vandermaaten.fit_d(self.DistanceMatrix,self.getParameters);
		end
	else
		% use RawData
		[R, cost] = TSNE.vandermaaten.fit(self.RawData', self.getParameters);

	end
elseif self.implementation == TSNE.implementation.multicore
	R = TSNE.multicore.fit(self.RawData)';
elseif self.implementation == TSNE.implementation.berman

	[R, cost] = TSNE.berman.fit(self.RawData','perplexity',self.perplexity);

else
	keyboard
end

% cache if need be
if self.UseCache
	save([cache_dir filesep H '.cache'],'R','cost','-nocompression','-v7.3')
end


