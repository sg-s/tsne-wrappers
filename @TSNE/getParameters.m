% returns a structure of parameters from the TSNE object

function p = getParameters(self)

p = struct;
fn = fieldnames(self);

for i = 1:length(fn)
	if strcmp(fn{i},'DistanceMatrix')
		continue
	elseif strcmp(fn{i},'RawData')
		continue
	elseif strcmp(fn{i},'implementation')
		continue
	end
	p.(fn{i}) = self.(fn{i});
end