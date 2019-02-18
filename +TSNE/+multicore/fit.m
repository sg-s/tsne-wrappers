% mctsne.m
% MATLAB wrapper for Dmitry Ulyanov's Multicore t-SNE 
% implementation
% this wrapper assumes you have the python wrapper set up
% and calls that.  

function R = fit(Vs,n_iter,perplexity, n_jobs)

if nargin < 2
	n_iter = 1000;
	perplexity = 30;
	n_jobs = mtools.core.numcores;
elseif nargin < 3
	perplexity = 30;
	n_jobs = mtools.core.numcores;
elseif nargin < 4
	n_jobs = mtools.core.numcores;
end


containing_dir = fileparts(which('TSNE.multicore.fit'));


save('Vs.mat','Vs','-v7.3')

perplexity = floor(perplexity);
n_iter = floor(n_iter);
assert(n_iter > 10,'n_iter too low')
assert(perplexity > 2,'perplexity too low')

p1 = ['python "' containing_dir];

% first check if the environment is right using the test script
e = system([p1 filesep 'mctsne_test.py" ']);
if e ~=0
	warning('MulticoreTSNE test failed...attempting to fix path')
	conda.setenv('mctsne')
end

p1 = ['"' containing_dir];
eval_str =  [p1 filesep 'mctsne.py" ' oval(perplexity) ' ' oval(n_iter) ' ' oval(n_jobs)];
system(eval_str)

% read the solution
R = h5read('data.h5','/R');

% clean up
delete('data.h5')
delete('Vs.mat')
