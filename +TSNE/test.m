% this tests the TSNE implementations

function test()

% download data if need be
[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;


% reshape data
X = reshape(imgDataTest,28*28,size(imgDataTest,4));
X = X(:,1:1e3); % only first 1000 points
labelsTest = double(labelsTest(1:1e3));

figure('outerposition',[300 300 1200 600],'PaperUnits','points','PaperSize',[1200 600]); hold on
subplot(1,2,1); hold on
title('MATLAB internal')

% test MATLAB's internal version
t = TSNE;
t.implementation = TSNE.implementation.internal;
t.raw_data = double(X');

R = t.fit;

c = lines;

for i = 1:10
	plot(R(labelsTest==i,1),R(labelsTest==i,2),'.','Color',c(i,:))
end




% test pairwise-distance tsne using synthetic data
X = [randn(333,1) randn(333,1); 10+randn(333,1) 3+randn(333,1); 6+randn(333,1) 9+randn(333,1)];
D = squareform(pdist(X));

t = TSNE;
t.implementation = TSNE.implementation.vandermaaten;
t.distance_matrix = D;

R = t.fit;

subplot(1,2,2); hold on
title('van der Maaten: distance matrix')
plot(R(1:333,1),R(1:333,2),'o')
plot(R(334:666,1),R(334:666,2),'o')
plot(R(667:end,1),R(667:end,2),'o')

