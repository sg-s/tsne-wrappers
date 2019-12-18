classdef TSNE < Hashable & Constructable 


properties

	implementation@TSNE.implementation = TSNE.implementation.vandermaaten
	DistanceMatrix@double
	RawData@double
	NumDims@double = 2
	perplexity@double = 30

	% cache embeddings to immediately recall it next time?
	UseCache@logical = true

	n_cores@double = corelib.numcores;

	NIter@double = 1e3;

	RandomSeed@double = 1984

	InitialSolution

	% degree of freedom parameter, controls substructure within a cluster
	% see Koback, Linderman, Steinerberger, Kluger & Berens for
	% an explanation of what this parameter is and why it's useful
	Alpha@double = 1;

	% tolerance in determining nearest neighbours
	Tolerance@double = 1e-5;

	InitialMomentum = .5;
	FinalMomentum = .8;

	 % iteration at which lying about P-values is stopped  
	StopLyingIter = 100;

	 % initial learning rate
	Epsilon = 500;

	% minimum gain for delta-bar-delta
	MinGain = .01; 

	% iteration at which momentum is changed
	MomSwitchIter = 250;                     
             
end



methods



	function self = TSNE(varargin)
		self = self@Constructable(varargin{:});   
	end



	function self = set.RawData(self, value)

		% should be a matrix
		assert(ismatrix(value),'RawData should be a matrix')

		% rotate correctly
		sz = size(value);
		if sz(1) > sz(2)
			value = value';
		end

		% check for NaNs and Infs
		assert(~any(isnan(sum(value))),'NaNs found in raw data. Use TSNE.clean() to clean it before setting')

		assert(~any(isinf(sum(value))),'Inf found in raw data. Use TSNE.clean() to clean it before setting')

		self.RawData = value;

	end

end


end % classdef