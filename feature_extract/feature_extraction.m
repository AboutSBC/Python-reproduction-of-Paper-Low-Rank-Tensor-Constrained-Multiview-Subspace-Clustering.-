%% path
path='coil-20-proc\'; 
num_view = 3;
nCls = 20;
num_sample = 1440;

%% extract feature
count = 1;

% data and ground truth
X = cell(  1, num_view );  
gt = zeros( num_sample, 1 );  

% each view, need to initialize the dimension of each view
intensity_feature_matrix = zeros(1024, num_sample);
LBP_feature_matrix = zeros(3304, num_sample);
Gabor_feature_matrix = zeros(6750, num_sample);

for i=1:20
    data_path = strcat(path, 'obj',int2str(i), '__');
    for j=0:71
        P=strcat(data_path,  int2str(j), '.png');

        % load image
        imag = imread(P); 
        
        %% Add noise
        GrayImage = im2gray(imag);
        GrayImage = im2double(GrayImage);

        %% get Feature
        % intensity
        intensity_feature_matrix(:, count) = extractintensity(GrayImage, [32 32]);
        
        % lbp
        mapping = getmapping(8,'u2');  % get mapping
        unit_norm = 'nh';  % no normalise
        LBP_feature_matrix(:, count) = extractlbp(GrayImage, [16 16],  [112 128], 1, 8, mapping, unit_norm);
        
        % gabor
        Gabor_feature_matrix(:, count) = extractgabor(GrayImage, [4], [0 45 90 135], [30 25], [90 75]);
        
        % gt
        gt(count) = i;
        
        count = count+1;
    end
end
X{1,1} = intensity_feature_matrix;
X{1,2} = LBP_feature_matrix;
X{1,3} = Gabor_feature_matrix;
save test_coil20.mat X gt  
