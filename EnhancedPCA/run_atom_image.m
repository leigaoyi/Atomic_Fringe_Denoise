clear;
[trainset_mat, trainset_avg_field, mask] = read_tif();
%load basis_mats.mat
load my_filts.mat

a = 3/16;
b = -13/16;

in_filter = my_filts;
filter_sz = size(in_filter);
big_filter = zeros(filter_sz(1),2*filter_sz(2));
big_filter(:,1:2:end) = real(in_filter);
big_filter(:,2:2:end) = imag(in_filter);

% input size
inL = 476;
maskR = 95; 

trainset = struct();
baseRoot = './data/atom_test/'
fileList=dir(fullfile(baseRoot));
for i = 3 :length(fileList);
    trainset.path{i-2} = [baseRoot fileList(i).name];
end

for i = 1: 100;
    atom_path = trainset.path{i}
    [a1, atom_name, atom_ext] = fileparts(atom_path);
    curr_im = double(imread(atom_path));
    curr_im = (curr_im/4294967295-b)/a;
    curr_im = exp(curr_im);
    %curr_im = curr_im * 255.;
    %curr_im = curr_im/4294967295*255;

    mask = mask(:); %% mask区域内为1
    %mask = logical(mask);
    curr_im_hot = reshape([curr_im], 476*476, 1);

    for i = 1:476*476;
        if mask(i)==0;
            curr_im_hot(i) = 0;
        end
    end

    mean_curr = mean(curr_im_hot);
    curr_im_hot = curr_im_hot - mean_curr - trainset_avg_field(:);
    %mask = 1-mask;


    %curr_mask = curr_im(mask,:) %% make the mask area 0


    %curr_im_rec = zeros(inL,inL);
    fit_coe = pinv(big_filter)*curr_im_hot;
    curr_im_rec = big_filter * fit_coe;
    curr_im_rec = reshape([curr_im_rec], 476, 476);
    curr_im_rec = curr_im_rec + mean_curr + trainset_avg_field;

    %curr_im_tif = ( curr_im_rec)/255;
    curr_im_tif = (log(curr_im_rec)*a + b) * 4294967295;

    %out_residual = curr_im-out_fit;

    % for j=1:30
    % %             recpnstruct the center
    %     w_0_j =  sum(curr_im.*big_filter(:,j));
    %     curr_im_rec = curr_im_rec +w_0_j*basis_mats(:,:,j);
    % end

    %         rescale the image
    % scale_fac = norm(curr_im(logical(1-mask)))/norm(curr_im_rec(logical(1-mask)));
    % curr_im_rec_scaled = curr_im_rec; %* scale_fac;
    % curr_im_real = curr_im_rec_scaled +mean_curr+trainset_avg_field(:);



    % curr_im_real_log = log(curr_im_real);
    % %curr_im_tif = (curr_im_real_log*a+b)*4294967295; % to 32 bit
    % curr_im_tif = (curr_im_real_log*a+b)*4294967295;

    outputPath_file = ['./result/',atom_name,'.tif'];
    t = Tiff(outputPath_file,'w');  %创建TIFF文件
    t.setTag('ImageLength', size(curr_im_tif,1));
    t.setTag('ImageWidth', size(curr_im_tif,2));
    t.setTag('Photometric', Tiff.Photometric.MinIsBlack);   %图像数据的颜色空间
    t.setTag('BitsPerSample', 32);                       %数据位数
    t.setTag('SamplesPerPixel', 1);
    t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
    t.setTag('Compression',Tiff.Compression.None);          %无压缩
    %t.setTag('SampleFormat',sf);                            %像素样本格式
    curr_im_tif = uint32(curr_im_tif);
    t.write(curr_im_tif);
    t.close;            %写完把文件关了

end
