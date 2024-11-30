% Define the paths for the three folders in the current directory
dir_path_ir = fullfile(pwd, 'ir');     
dir_path_vis = fullfile(pwd, 'vis');     
dir_path_fused = fullfile(pwd, 'fuse');  


% Get the .jpg and .png image files in each folder
dir_source_ir_jpg = dir(fullfile(dir_path_ir, '*.jpg'));    % Get .jpg format infrared images
dir_source_ir_png = dir(fullfile(dir_path_ir, '*.png'));    % Get .png format infrared images
dir_source_ir = [dir_source_ir_jpg; dir_source_ir_png];     % Combine both file formats

dir_source_vis_jpg = dir(fullfile(dir_path_vis, '*.jpg'));  % Get .jpg format visible light images
dir_source_vis_png = dir(fullfile(dir_path_vis, '*.png'));  % Get .png format visible light images
dir_source_vis = [dir_source_vis_jpg; dir_source_vis_png];  % Combine both file formats

dir_fused_jpg = dir(fullfile(dir_path_fused, '*.jpg'));     % Get .jpg format fused images
dir_fused_png = dir(fullfile(dir_path_fused, '*.png'));     % Get .png format fused images
dir_fused = [dir_fused_jpg; dir_fused_png];                 % Combine both file formats

% Get the number of images
num_ir = length(dir_source_ir);
disp(['Number of infrared images: ', num2str(num_ir)]);
num_vis = length(dir_source_vis);
disp(['Number of visible light images: ', num2str(num_vis)]);
num_fused = length(dir_fused);
disp(['Number of fused images: ', num2str(num_fused)]);

% Initialize the results array
results = zeros(1, 4);  % [EN, MI, SCD, MS_SSIM]

for i = 1:num_ir

    % Get the base name of the ir file, removing the file extension
    [~, base_name, ~] = fileparts(dir_source_ir(i).name);  % For example, 'testIR_00000'

    % Display the currently processed file name for debugging
    disp(['Processing IR file: ', base_name]);

    % Create the pattern to match the fused image name, supporting .jpg and .png formats
    fuse_name_pattern_jpg = ['fused_rfnnest_700_wir_6.0_wvi_3.0axial_' base_name '.jpg'];
    fuse_name_pattern_png = ['fused_rfnnest_700_wir_6.0_wvi_3.0axial_' base_name '.png'];

    % Find the corresponding filename in the fuse folder
    fuse_match_jpg = strcmp({dir_fused.name}, fuse_name_pattern_jpg);
    fuse_match_png = strcmp({dir_fused.name}, fuse_name_pattern_png);
    
    if sum(fuse_match_jpg) == 0 && sum(fuse_match_png) == 0
        error(['No matching fused image found for ' base_name]);
    elseif sum(fuse_match_jpg) > 0
        fused_file = dir_fused(fuse_match_jpg);  % Match .jpg file
    else
        fused_file = dir_fused(fuse_match_png);  % Match .png file
    end
    
    % Display the matched fused filename for debugging
    disp(['Matched Fused file: ', fused_file.name]);
    
    % Read images
    fused_image = imread(fullfile(dir_path_fused, fused_file.name));
    source_image1 = imread(fullfile(dir_path_ir, dir_source_ir(i).name));
    source_image2 = imread(fullfile(dir_path_vis, dir_source_vis(i).name));

    % Check if images were successfully read
    if isempty(fused_image) || isempty(source_image1) || isempty(source_image2)
        error('Image reading failed, please check the image files and paths');
    end

    % Display the dimensions of the read images for debugging
    disp(['Fused image size: ', num2str(size(fused_image))]);
    disp(['Source image1 (IR) size: ', num2str(size(source_image1))]);
    disp(['Source image2 (VIS) size: ', num2str(size(source_image2))]);

    % If source_image1 and source_image2 are RGB color images, convert them to grayscale
    if size(source_image1, 3) == 3
        source_image1 = rgb2gray(source_image1);
    end
    if size(source_image2, 3) == 3
        source_image2 = rgb2gray(source_image2);
    end
    
    % Ensure consistent dimensions, resize if necessary
    if size(source_image1) ~= size(fused_image)
        source_image1 = imresize(source_image1, [size(fused_image, 1), size(fused_image, 2)]);
    end
    if size(source_image2) ~= size(fused_image)
        source_image2 = imresize(source_image2, [size(fused_image, 1), size(fused_image, 2)]);
    end

    % Call the analysis function to get metrics
    [EN, MI, SCD, MS_SSIM] = analysis_Reference(fused_image, source_image1, source_image2);

    % Check for NaN values in the return values and display debugging info
    if any(isnan([EN, MI, SCD, MS_SSIM]))
        warning(['Analysis function returned NaN values, file: ' fused_file.name]);
        disp(['EN: ', num2str(EN), ', MI: ', num2str(MI), ', SCD: ', num2str(SCD), ', MS_SSIM: ', num2str(MS_SSIM)]);
    else
        % Accumulate all metrics
        results = results + [EN, MI, SCD, MS_SSIM];
    end
end

% Display the average values
disp(results / num_ir);
disp('Done');
