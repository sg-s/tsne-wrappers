% plots the t-sne embedding in 2D
function plot(self, ax, labels)

R = self.fit();



if nargin < 2
	figure('outerposition',[300 300 600 600],'PaperUnits','points','PaperSize',[1200 600]); hold on
	ax = gca;
end


if exist('labels','var')
	unique_labels = unique(labels);
	n_labels = length(unique_labels);
	c = parula(n_labels);

	for i = 1:n_labels
		plot(ax,R(labels==unique_labels(i),1),R(labels==unique_labels(i),2),'.','Color',c(i,:))
	end


else
	plot(ax,R(:,1),R(:,2),'k.')
end

