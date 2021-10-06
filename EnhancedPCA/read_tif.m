function [trainset_mat, trainset_avg_field, mask] = read_tif()

% number of images to be considered for the PCA
PCA_num_images = 54;

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


trainset = struct();
baseRoot = './basis/'
fileList=dir(fullfile(baseRoot));
for i = 3 :length(fileList);
    trainset.path{i-2} = [baseRoot fileList(i).name];
end


trainset_mat = zeros(inL,inL,PCA_num_images);
trainset_avg_field =zeros(inL,inL);
u_col = zeros(num_p_unmasked,PCA_num_images);

for i=1:PCA_num_images
%     load the file & retrieve the original absorption image
        curr_im_path = trainset.path{i};
        curr_im = double(imread(curr_im_path));
        curr_im = (curr_im/4294967295-b)/a;
        curr_im = exp(curr_im);
        %curr_im = curr_im ;
        %curr_im = curr_im/4294967295*255;

%         Save the train image to mat stack
        trainset_avg(i) = mean(curr_im(:));
        curr_im = curr_im - trainset_avg_field - trainset_avg(i);
        trainset_mat(:,:,i) = curr_im;

%         calculate average light intensity
        trainset_avg_field = trainset_avg_field + curr_im;
end

% subtract average intensity & save it for future reference
%trainset_avg_field = trainset_avg_field/height(trainset);
trainset_avg_field = trainset_avg_field/PCA_num_images;

trainset_mat = trainset_mat - repmat(trainset_avg_field,1,1,PCA_num_images);
    
    