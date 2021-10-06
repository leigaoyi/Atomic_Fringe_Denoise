function [out_fd, out_filter] = create_filter(in_data, in_pix)

in_fft = fftshift(fft2(in_data));
abs_fft = abs(in_fft);
[m, p] = max(abs_fft);
%in_pix = p

[max_loc] = maxnd(abs(in_fft));

my_sz = size(in_fft);

[c_basis_x,c_basis_y] = ndgrid(1:my_sz(1),1:my_sz(2));

out_filter = exp(-(c_basis_x-max_loc(1)).^2*in_pix^2/(2*my_sz(1)^2)-(c_basis_y-max_loc(2)).^2*in_pix^2/(2*my_sz(2)^2));

out_fd = ifft2(ifftshift(in_fft.*out_filter));
out_fd = reshape(out_fd,my_sz(1)*my_sz(2),[]);