% embed data or distance matrix
% 
function [R, cost] = fit(self)


start_iter = 1;

% first check the hash
if self.UseCache

	% don't hash NIter
	NIter = self.NIter;
	self.NIter = 1;
	H = self.hash;
	self.NIter = NIter;
	
	cache_dir = fileparts(fileparts(which('TSNE.implementation')));
	cache_dir = [cache_dir filesep '.cache'];
	if ~isdir(cache_dir)
		mkdir(cache_dir)
	end

	if exist([cache_dir filesep H '.cache'],'file') == 2 && self.UseCache
		disp('Using cached solution...')

		load([cache_dir filesep H '.cache'],'R','cost','iter','-mat');

		if iter < self.NIter
			% need to keep going
			self.InitialSolution = R;
			start_iter = iter;
		else
			self.NIter = iter;
			return
		end


	end
end

disp('Cache miss')



if self.implementation == TSNE.implementation.internal
	[R, cost] = tsne(self.RawData');

elseif self.implementation == TSNE.implementation.vandermaaten
	if isempty(self.RawData) && ~isempty(self.DistanceMatrix)
		% use distance matrix
		[R, cost] = TSNE.vandermaaten.fit_d(self.DistanceMatrix,self.getParameters, H, start_iter);

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
	iter = self.NIter;
	save([cache_dir filesep H '.cache'],'R','cost','iter','-nocompression','-v7.3')
end


