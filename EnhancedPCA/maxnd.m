function [coords, max_val] = maxnd(in_data)
[max_val, max_loc] = max(in_data(:));
[coords(1), coords(2), coords(3), coords(4)] = ind2sub(size(in_data),max_loc);
coords = coords(1:numel(size(in_data)));
end