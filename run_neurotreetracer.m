function run_neurotreetracer(inputDir, outputDir, tracerDir)
% RUN_NEUROTREETRACER  Batch driver for NeuroTreeTracer tree extraction.
%
%   run_neurotreetracer(inputDir, outputDir, tracerDir)
%
%   inputDir  : folder holding inputSeg.mat and inputSoma.mat, as written by
%               neurotreetracer_export_crop.py
%   outputDir : folder to write results into (created if missing)
%   tracerDir : clone of github.com/cihanbilge/AutomatedTreeStructureExtraction
%
% This is a batch replacement for the repository's Script.m. Two differences
% from the shipped script matter:
%
%   option.manual = 0   Script.m ships with manual=1, which calls input() and
%                       imshow() for every single neurite and blocks forever in
%                       a non-interactive session. Batch runs must use 0.
%   -v7 save format     scipy.io.loadmat cannot read MATLAB's -v7.3 (HDF5)
%                       files. Everything here is saved as -v7 so the Python
%                       importer can read it.
%
% Besides the raw neuriteGraph cell array, a flat [somaId neuriteId linearIndex]
% matrix is written, so the importer never has to parse nested cell arrays.

if nargin < 3
    error('Usage: run_neurotreetracer(inputDir, outputDir, tracerDir)');
end

addpath(tracerDir);
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

segFile  = fullfile(inputDir, 'inputSeg.mat');
somaFile = fullfile(inputDir, 'inputSoma.mat');
if ~exist(segFile, 'file'),  error('Missing %s', segFile);  end
if ~exist(somaFile, 'file'), error('Missing %s', somaFile); end

S = load(segFile);   inputSeg  = S.inputSeg;
S = load(somaFile);  inputSoma = S.inputSoma;

inputSeg  = double(inputSeg  > 0);
inputSoma = double(inputSoma > 0);
sz = size(inputSeg);

nSomas = max(max(bwlabel(inputSoma > 0)));
fprintf('Input     : %d x %d, %d soma components\n', sz(1), sz(2), nSomas);
fprintf('Rectangles: %d full-size masks, about %.1f GB\n', ...
        18*10*360, 18*10*360*sz(1)*sz(2)/1e9);

option.rect   = 1;   % generate the seed-search rectangles (expensive, see above)
option.manual = 0;   % fully automated; never prompt

tic;
[neuriteGraph, cell_rect] = runCenterLineParallel(inputSeg, inputSoma, option); %#ok<ASGLU>
elapsed = toc;
fprintf('Tracing finished in %.1f s (%.1f min)\n', elapsed, elapsed/60);

% Flatten to [somaId neuriteId linearIndex]. neuriteGraph is allocated in
% generateCenterLine.m as cell(comp_Num), i.e. comp_Num x comp_Num, and only the
% first column is filled - so iterate 1:nSomas, not numel(neuriteGraph).
traces = zeros(0, 3);
nTraced = 0;
for i = 1:nSomas
    if i > numel(neuriteGraph), break; end
    somaTraces = neuriteGraph{i};
    if isempty(somaTraces), continue; end
    if ~iscell(somaTraces), somaTraces = {somaTraces}; end
    for j = 1:numel(somaTraces)
        p = somaTraces{j};
        if isempty(p), continue; end
        p = double(p(:));
        traces = [traces; repmat(i, numel(p), 1), repmat(j, numel(p), 1), p]; %#ok<AGROW>
        nTraced = nTraced + 1;
    end
end
fprintf('Traced %d neurites over %d somas, %d trace points\n', ...
        nTraced, nSomas, size(traces, 1));

imageSize   = sz;                          %#ok<NASGU>
elapsedSec  = elapsed;                     %#ok<NASGU>
somaCount   = nSomas;                      %#ok<NASGU>
neuriteCount= nTraced;                     %#ok<NASGU>

save(fullfile(outputDir, 'traces_flat.mat'), ...
     'traces', 'imageSize', 'elapsedSec', 'somaCount', 'neuriteCount', '-v7');
save(fullfile(outputDir, 'neuriteGraph_raw.mat'), 'neuriteGraph', '-v7');

fprintf('Wrote %s\n', fullfile(outputDir, 'traces_flat.mat'));
end
