# tsne-wrappers

MATLAB wrappers for many commonly used t-SNE implementations. This is a work in progress. Do not use. 

| Implementation | Support | Notes |
| -------------- | -------- | ------- |
| [Multicore-tSNE](https://github.com/DmitryUlyanov/Multicore-TSNE) | ✅ | uses [my branch](https://github.com/sg-s/Multicore-TSNE), which works as intended on macOS |
| internal | ✅  | MATLAB's internal implementation | 
| [van der Maaten](https://lvdmaaten.github.io/tsne/) | 🚧 | allows embedding from pairwise distances |
| [FI-tSNE](https://github.com/KlugerLab/FIt-SNE) | 🚧  | |
| [Gordon Berman's t-SNE](https://github.com/gordonberman/MotionMapper) | 🚧  | used in Motion Mapper |


## Changes to published implementations 

### van der Maaten's implementation

* estimation of probabilites (`d2p.m`) can run in parallel, which means it can be much faster than the single-threaded implementation in the original 
* contains an assertion to prevent NaNs from propagating through affinity matrix; instead throws an error suggesting a fix (increase perplexity)