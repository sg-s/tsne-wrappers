% this tests the TSNE implementations

function test(tests_to_run)


if strcmp(tests_to_run,'all') || strcmp(tests_to_run,'mnist') 

	% download data if need be
	[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = TSNE.internal.prepareData;


	% reshape data
	X = reshape(imgDataTest,28*28,size(imgDataTest,4));
	X = X(:,1:1e3); % only first 1000 points
	labelsTest = double(labelsTest(1:1e3));

	figure('outerposition',[300 300 1200 901],'PaperUnits','points','PaperSize',[1200 901]); hold on
	prettyFig();



	% compare different implemntations 


	ax = subplot(2,3,1); hold on
	title(ax, 'MATLAB internal')
	t = TSNE;
	t.implementation = TSNE.implementation.internal;
	t.raw_data = double(X);
	t.plot(ax,labelsTest);
	axis off
	drawnow

	ax = subplot(2,3,2); hold on
	title(ax, 'multicore')
	t = TSNE;
	t.implementation = TSNE.implementation.multicore;
	t.raw_data = double(X);
	t.plot(ax,labelsTest);
	axis off
	drawnow


	ax = subplot(2,3,3); hold on
	title(ax, 'van der Maaten')
	t = TSNE;
	t.implementation = TSNE.implementation.vandermaaten;
	t.raw_data = double(X);
	t.plot(ax,labelsTest);
	axis off
	drawnow

	ax = subplot(2,3,4); hold on
	title(ax, 'Berman')
	t = TSNE;
	t.implementation = TSNE.implementation.berman;
	t.raw_data = double(X);
	t.plot(ax,labelsTest);
	axis off
	drawnow
end


if strcmp(tests_to_run,'all') || strcmp(tests_to_run,'distance_matrix') 

	% use frozen random numbers
	RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1984)); 

	% test pairwise-distance tsne using synthetic data
	X = [randn(333,1) randn(333,1); 10+randn(333,1) 3+randn(333,1); 6+randn(333,1) 9+randn(333,1)];
	D = squareform(pdist(X));

	t = TSNE;
	t.use_cache = false;
	t.implementation = TSNE.implementation.vandermaaten;
	t.distance_matrix = D;

	R = t.fit;

	subplot(1,2,2); hold on
	title('van der Maaten: distance matrix')
	plot(R(1:333,1),R(1:333,2),'o')
	plot(R(334:666,1),R(334:666,2),'o')
	plot(R(667:end,1),R(667:end,2),'o')

	GetMD5(R)

end