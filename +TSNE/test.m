% this tests the TSNE implementations

function test(tests_to_run)


if strcmp(tests_to_run,'all') || strcmp(tests_to_run,'mnist') 

	% download data if need be
	[~, ~, imgDataTest, labelsTest] = TSNE.internal.prepareData;


	% reshape data
	X = reshape(imgDataTest,28*28,size(imgDataTest,4));
	X = X(:,1:1e3); % only first 1000 points
	labelsTest = double(labelsTest(1:1e3));

	figure('outerposition',[300 300 1200 901],'PaperUnits','points','PaperSize',[1200 901]); hold on
	figlib.pretty();



	% make a colormap
	C = lines(10);
	C(8,:) = [0 0 0];
	C(9,:) = [1 0 0];
	C(10,:) = [0 0 1];

	% compare different implemntations 


	ax = subplot(2,3,1); hold on
	title(ax, 'MATLAB internal')
	t = TSNE;
	t.implementation = TSNE.implementation.internal;
	t.RawData = double(X);
	t.plot(ax,labelsTest,C);
	axis off
	drawnow



	% ax = subplot(2,3,2); hold on
	% title(ax, 'multicore')
	% t = TSNE;
	% t.implementation = TSNE.implementation.multicore;
	% t.RawData = double(X);
	% t.plot(ax,labelsTest<C);
	% axis off
	% drawnow


	ax = subplot(2,3,3); hold on
	title(ax, 'van der Maaten')
	t = TSNE;
	t.implementation = TSNE.implementation.vandermaaten;
	t.RawData = double(X);
	t.plot(ax,labelsTest,C);
	axis off
	drawnow

	ax = subplot(2,3,4); hold on
	title(ax, 'Berman')
	t = TSNE;
	t.implementation = TSNE.implementation.berman;
	t.RawData = double(X);
	t.plot(ax,labelsTest);
	axis off
	drawnow
end


if strcmp(tests_to_run,'all') || strcmp(tests_to_run,'distance_matrix') 

	% use frozen random numbers
	%RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1984)); 

	% test pairwise-distance tsne using synthetic data
	X = [randn(333,1) randn(333,1); 10+randn(333,1) 3+randn(333,1); 6+randn(333,1) 9+randn(333,1)];
	D = squareform(pdist(X));

	t = TSNE;
	t.UseCache = false;
	t.implementation = TSNE.implementation.vandermaaten;
	t.DistanceMatrix = D;

	R = t.fit;

	subplot(1,2,2); hold on
	title('van der Maaten: distance matrix')
	plot(R(1:333,1),R(1:333,2),'o')
	plot(R(334:666,1),R(334:666,2),'o')
	plot(R(667:end,1),R(667:end,2),'o')

	mtools.crypto.md5hash(R)

end


% effect of changing the Alpha parameter in the genralized t-distribution
% to reveal substructure of data
if strcmp(tests_to_run,'all') || strcmp(tests_to_run,'finer-tsne') 

	% use frozen random numbers
	RandStream.setGlobalStream(RandStream('mt19937ar','Seed',0)); 

	% create 3 Gaussian clusters equally spaced away from each other
	Sigma = .001;
	N = 200;
	Offset = 0.05;
	XY1 = mvnrnd([-1 0],[Sigma Sigma],N);
	XY1 = [XY1; mvnrnd([-1 + Offset 0],[Sigma Sigma],N)];

	XY2 = mvnrnd([1 0],[Sigma Sigma],N);
	XY2 = [XY2; mvnrnd([1 + Offset 0],[Sigma Sigma],N)];

	XY3 = mvnrnd([0 sqrt(3)],[Sigma Sigma],N);
	XY3 = [XY3; mvnrnd([0 Offset + sqrt(3)],[Sigma Sigma],N)];

	XY = [XY1; XY2; XY3];

	L = repmat(1:3,N*2,1);
	L = L(:);

	C = lines;


	figure('outerposition',[300 300 1200 1111],'PaperUnits','points','PaperSize',[1200 1111]); hold on
	subplot(2,2,1); hold on
	for i = 1:3
		plot(XY(L==i,1),XY(L==i,2),'.')
	end
	title('Raw Data')
	axis square
	axis off


	figlib.pretty()
	

	t = TSNE('RawData',XY);
	t.Alpha = 1;

	subplot(2,2,2); hold on
	title('Alpha = 1')
	t.plot(gca,L,C);
	axis off

	t.Alpha = .8;
	subplot(2,2,3); hold on
	title('Alpha = .5')
	t.plot(gca,L,C);
	axis off


	t.Alpha = 100;
	subplot(2,2,4); hold on
	title('Alpha = 100')
	t.plot(gca,L,C);
	axis off


end