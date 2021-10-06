clear;
[trainset_mat, trainset_avg_field, mask] = read_tif();

num_rounds = 15;
num_vecs = 4;
basic_size = size(trainset_mat);

% mask = mask(:);
% trainset_mat = reshape([trainset_mat], 476*476, basic_size(3));
% trainset_mat = trainset_mat(logical(mask));

my_filts = zeros(basic_size(1)*basic_size(2),num_rounds*(num_vecs+1)+num_vecs); 
%trainset_mat = reshape([trainset_mat], basic_size(1)*basic_size(2), basic_size(3));

all_lights = trainset_mat;
mean_light = squeeze(mean(all_lights,3));
all_lights_out = all_lights-mean_light;

all_lights = reshape(all_lights_out,basic_size(1)*basic_size(2), basic_size(3));

all_lights_orig = all_lights;

for round = 1:num_rounds;
	base_size = (round-1)*(num_vecs+1);
    if round > 1;
		my_fits = fit_with_filter_new(my_filts(:,1:(base_size+1)),all_lights_orig);
		all_lights = all_lights_orig-my_fits;
	end
	
	for a = (base_size+1):(base_size+2*num_vecs);
		corr_mtx = all_lights'*all_lights;
		fprintf('Residual signal: %.6g\n', trace(corr_mtx));
		[cevec, ~] = eigs(corr_mtx,1);
		fevec = all_lights*cevec;
		fevec = reshape(fevec,basic_size(1),basic_size(2));
		my_filts(:,a) = create_filter(fevec,30);
		my_fits = fit_with_filter_new(my_filts(:,1:a),all_lights_orig);
		
		all_lights = all_lights_orig-my_fits;
	end
	corr_mtx = all_lights'*all_lights;
	[cevec, ~] = eigs(corr_mtx,1);
	fevec = all_lights*cevec;
	my_filts(:,a+1) = fevec;
	
	my_fit_coeffs = my_filts(:,1:(a+1))\all_lights_orig;
	corr_mtx = my_fit_coeffs*my_fit_coeffs';
	[cevec, ~] = eig(corr_mtx,'vector');
	fevec = my_filts(:,1:(a+1))*cevec;
	my_filts(:,1:(size(fevec,2)-num_vecs)) = fevec(:,(num_vecs+1):end);
	
	fprintf('Round %d complete.\n',round);
end

my_filts = my_filts(:,1:(size(fevec,2)-num_vecs));
save my_filts.mat my_filts
