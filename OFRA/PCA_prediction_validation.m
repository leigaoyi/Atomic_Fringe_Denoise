%% Constants & flags
clear

% number of images to be considered for the PCA
PCA_num_images = 600;

% normalization constants for the logarithmic image
a = 3/16;
b = -13/16;

% input size
inL = 476;
maskR = 95; 
center_x = inL/2; center_y = inL/2; 

% logical mask
mask = zeros(inL,inL);
for i = 1:inL
    for j = 1:inL
        if sqrt((i-center_x)^2+(j-center_x)^2) > maskR
            mask(i,j) = 1;
        end
    end
end

% number of unmasked pixels
inv_mask = 1-mask;
num_p_unmasked = inL^2-sum(inv_mask(:));

% flags
calc_new_PCA_basis = 0; % =1 if you wish to construct a new basis, =0 if you wish to use the latest basis 
calc_new_PCA_basis_mats = 0;
%% PCA training dataset
if calc_new_PCA_basis == 1
    fid = fopen('firstDayTrain.csv');
    trainset_full = textscan(fid, '%s', 'delimiter','\n');
    fclose(fid);
    trainset_full = table(trainset_full{1,1}, 'VariableNames', {'path'});

    % randomly chose n images for the PCA protocol
    [trainset,idx] = datasample(trainset_full,PCA_num_images,'Replace',false);
    trainset_leftover = setdiff(trainset_full,trainset);

    save trainset_PCA.mat trainset
    save trainset_LeftOver_PCA.mat trainset_leftover
else
    load trainset_PCA.mat
    load trainset_LeftOver_PCA.mat
end
%% Load and center the trainset
datasets_main_dir = 'C:/datasets/single_shot_a3_16_b_m13_16_32bit/';
% datasets_main_dir_multi = '//132.68.68.10/C$/datasets/single_shot_a3_16_b_m13_16_32bit/'; 

if calc_new_PCA_basis == 1
    trainset_mat = zeros(inL,inL,PCA_num_images);
    trainset_avg_field =zeros(inL,inL);
    u_col = zeros(num_p_unmasked,PCA_num_images);

    for i=1:PCA_num_images
    %     load the file & retrieve the original absorption image
            curr_im_path = trainset.path{i};
            curr_im = double(imread(curr_im_path));
            curr_im = (curr_im/4294967295-b)/a;
            curr_im = exp(curr_im);

    %         Save the train image to mat stack
            trainset_avg(i) = mean(curr_im(:));
            curr_im = curr_im - trainset_avg_field - trainset_avg(i);
            trainset_mat(:,:,i) = curr_im;

    %         calculate average light intensity
            trainset_avg_field = trainset_avg_field + curr_im;
    end

    % subtract average intensity & save it for future reference
    trainset_avg_field = trainset_avg_field/height(trainset);
    trainset_mat = trainset_mat - repmat(trainset_avg_field,1,1,PCA_num_images);
    save trainset_avg_field.mat trainset_avg_field
    % transform to a column matrix
    for i=1:PCA_num_images
        ful_im = trainset_mat(:,:,i);
        edge_pixels = ful_im(logical(mask));
        u_col(:,i) = reshape(edge_pixels,[num_p_unmasked,1]);
    end
end
%% PCA

if calc_new_PCA_basis==1
    
    [eigvec,eigvals,k] = svd(u_col,'econ');
    
    eigvals_vec = diag(eigvals);
    save eigvec.mat eigvec
    save eigvals_vec.mat eigvals_vec
    
    if calc_new_PCA_basis_mats == 1
        basis_mats = zeros(inL,inL,PCA_num_images);
        for j = 1:PCA_num_images
            disp(['j=',num2str(j)])
            v_j = eigvec(:,j);
            C(:,j) = u_col\v_j;  %#ok<SAGROW>
            for i= 1:PCA_num_images
                disp(['i=',num2str(i)])
                basis_mats(:,:,j) = basis_mats(:,:,j) + trainset_mat(:,:,i)*C(i,j);
            end
        end
    end
     save basis_mats.mat basis_mats
else
    load  eigvec.mat 
    load trainset_avg_field.mat
    load basis_mats.mat
end
%% Chosing number of images for reconstruction

im_num_rec = 250;
%% Validation OD image reconstruction

datasetsMainDir = 'C:\datasets\single_shot_a3_16_b_m13_16_32bit\';
predictToSuffix = '_PCApredicted';
datasets = {'A_no_atoms_validation_cropped',...
    'A_with_atoms', 'R_leftovers_validation_cropped',...
    'R_no_atoms_validation_cropped', 'R_with_atoms_validation_cropped'};

for pathIdx = 1:length(datasets)
    disp(['pathIdx=',num2str(pathIdx)]);
    inputPath = [datasetsMainDir datasets{pathIdx}];
    outputPath = [strrep(inputPath, 'validation_cropped', 'val') predictToSuffix];
    mkdir(outputPath)
    dirs = dir(inputPath);
    images_names = rmfield(dirs,{'folder','date','bytes','isdir','datenum'});

    if isempty(images_names) == 1
        continue
    end
    images_names(1:2,:) = [];
    images_names = struct2cell(images_names)';
    
    for i=1:length(images_names)
%         load and process image
        val_image_large_dir = [strrep(inputPath, '_validation_cropped', ''),'\', char(images_names(i,1))];
        val_image_large_dir = strrep(val_image_large_dir,'bin','tif');
        
        curr_im = double(imread(val_image_large_dir));
        curr_im = (curr_im/4294967295-b)/a;
        curr_im = exp(curr_im);
        mean_curr_im = mean(curr_im(:));
        curr_im = curr_im - mean_curr_im;
        curr_im_masked = curr_im(logical(mask));
        curr_im_col = reshape(curr_im_masked,[num_p_unmasked,1]);
        curr_im_rec = zeros(inL,inL);
        
        for j=1:im_num_rec
%             recpnstruct the center
            w_0_j =  sum(curr_im_col.*eigvec(:,j));
            curr_im_rec = curr_im_rec +w_0_j*basis_mats(:,:,j);
        end
        
%         rescale the image
        scale_fac = norm(curr_im(logical(mask)))/norm(curr_im_rec(logical(mask)));
        curr_im_rec_scaled = curr_im_rec*scale_fac;
        curr_im_real = curr_im_rec_scaled +trainset_avg_field  + mean_curr_im ;
        curr_im_real_log = log(curr_im_real);
        outputPath_file = [outputPath,'\',char(images_names(i,1))];
        outputPath_file = strrep(outputPath_file ,'bin','mat');
        save(outputPath_file,'curr_im_real_log');
    end
end

%% Validation OD image reconstruction for trainset leftovers

datasetsMainDir = 'C:/datasets/single_shot_a3_16_b_m13_16_32bit/';
% datasets_main_dir_multi = '//132.68.68.10/C$/datasets/single_shot_a3_16_b_m13_16_32bit/'; 

dirD1val = [datasetsMainDir,'predictedPCAval/'];


mkdir(dirD1val);
mkdir([dirD1val,'A_no_atoms']);
mkdir([dirD1val,'R_no_atoms']);
mkdir([dirD1val,'R_with_atoms']);
mkdir([dirD1val,'R_leftovers']);


for i=1:size(trainset_leftover,1)
%         load and process image
    
    leftover_image_inpath = trainset_leftover.path{i};
    curPredPath = strrep(leftover_image_inpath, datasetsMainDir , dirD1val);
    leftover_image_outpath = strrep(curPredPath ,'tif','mat');

    curr_im = double(imread(leftover_image_inpath));
    curr_im = (curr_im/4294967295-b)/a;
    curr_im = exp(curr_im);
    mean_curr_im = mean(curr_im(:));
    curr_im = curr_im - mean_curr_im;
    curr_im_masked = curr_im(logical(mask));
    curr_im_col = reshape(curr_im_masked,[num_p_unmasked,1]);
    curr_im_rec = zeros(inL,inL);

    for j=1:im_num_rec
%             recpnstruct the center
        w_0_j =  sum(curr_im_col.*eigvec(:,j));
        curr_im_rec = curr_im_rec +w_0_j*basis_mats(:,:,j);
    end

%         rescale the image
    scale_fac = norm(curr_im(logical(mask)))/norm(curr_im_rec(logical(mask)));
    curr_im_rec_scaled = curr_im_rec*scale_fac;
    curr_im_real = curr_im_rec_scaled +trainset_avg_field  + mean_curr_im ;
    curr_im_real_log = log(curr_im_real);
    save(leftover_image_outpath,'curr_im_real_log');
    
end
