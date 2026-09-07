function CH = bwconvhull(BW)
% BWCONVHULL  Octave replacement for the MATLAB Image Processing Toolbox function.
%
%   CH = bwconvhull(BW) returns the convex hull of all foreground pixels of BW,
%   matching MATLAB's default 'union' method.
%
% Why this file exists: NeuroTreeTracer calls bwconvhull in getRectangle_alt.m,
% and the Octave image package does not provide it. There it is only ever used to
% turn the six marked corner points of a rectangle into a filled rectangle.
%
% Performance note: createRectangles calls this 18*10*360 = 64800 times. A naive
% implementation that runs inpolygon over the full image grid costs O(H*W) per
% call and made a 96x96 run take ~290 s. The hull can only cover the bounding box
% of its own points, so inpolygon is evaluated on that box alone and the result is
% pasted back. Semantics are unchanged; only the wasted work outside the box goes.
%
% Put this directory on the Octave path BEFORE the tracer directory:
%   addpath('octave_shims'); addpath('external/NeuroTreeTracer');

  BW = logical(BW);
  CH = false(size(BW));
  [r, c] = find(BW);

  if isempty(r)
    return;
  end
  if numel(r) < 3
    CH(BW) = true;
    return;
  end

  % Collinear iff the centred coordinate matrix has rank 1, i.e. its second
  % singular value vanishes. Comparing cross products against a single fixed
  % reference point is not safe: that point can itself sit at the centroid.
  % Checked up front because qhull prints a long precision-error block to
  % stderr before raising, and this runs 64800 times.
  M = [c(:) - mean(c), r(:) - mean(r)];
  s = svd(M);
  if numel(s) < 2 || s(2) <= 1e-9 * max(s(1), 1)
    CH(BW) = true;   % a line (or a point) is its own convex hull
    return;
  end

  try
    k = convhull(c, r);
  catch
    CH(BW) = true;
    return;
  end

  % Restrict the point-in-polygon test to the hull's own bounding box.
  r0 = max(1, floor(min(r)));  r1 = min(size(BW, 1), ceil(max(r)));
  c0 = max(1, floor(min(c)));  c1 = min(size(BW, 2), ceil(max(c)));

  [cc, rr] = meshgrid(c0:c1, r0:r1);
  inside = inpolygon(cc, rr, c(k), r(k));

  CH(r0:r1, c0:c1) = logical(inside);
  CH(BW) = true;   % never lose an input pixel to boundary rounding
end
