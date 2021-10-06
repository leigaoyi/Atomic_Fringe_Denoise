function [out_fit, out_residual, out_resnorm] = fit_with_filter_new(in_filter, in_raw, in_block_area)

filter_sz = size(in_filter);
big_filter = zeros(filter_sz(1),2*filter_sz(2));
big_filter(:,1:2:end) = real(in_filter);
big_filter(:,2:2:end) = imag(in_filter);

if nargin >= 3 
	mask = true(in_block_area(1),in_block_area(2));
	mask(in_block_area(4):in_block_area(6),in_block_area(3):in_block_area(5)) = false;
	mask = mask(:);
else
	mask = true(1,size(in_raw,1));
end
mask_filter = big_filter(mask,:);
mask_raw = in_raw(mask,:);

filt_coeffs = pinv(mask_filter)*mask_raw;

out_fit = big_filter*filt_coeffs;

if nargout >= 2
	out_residual = in_raw-out_fit;
	if nargout >= 3
		out_resnorm = sum(out_residual(mask).^2);
	end
end