clear all 
close all
%% OS DETECTION
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
    sepEnvironment = ':';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
    sepEnvironment = ';';
else
    disp('OS not compatible');
    return;
end
%% CONFIGURE PATHS
% APIRL PATH
apirlPath = 'D:\Martin\apirl-code\';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);

outputPath =  '../../../data/BigBrainSimulations/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
phantomOutputPath = [outputPath 'Phantoms/'];
if ~isdir(phantomOutputPath)
    mkdir(phantomOutputPath)
end

%% BIG BRAIN PHANTOM
% You can download the data from https://doi.org/10.5281/zenodo.1190598 
bigBrainPhantomFilename = '../../../data/phantom_atlas_density_cls.mat';
%% INIT CLASS GPET FOR HIGH RESOLUTION SIMULATION
fwhm_psf_phys_mm = 2.56;
fwhm_psf_all_mm = 4.3;
resolutionScaleFactor = 2;
PETsimu.scanner = 'mMR';
PETsimu.method =  'otf_siddon_gpu';
PETsimu.random_algorithm = 'from_ML_singles_matlab';
PETsimu.PSF.type = 'shift-invar';
PETsimu.PSF.Width = fwhm_psf_all_mm;
PETsimu = classGpet(PETsimu);
paramPETsimu.image_size.matrixSize = PETsimu.image_size.matrixSize * resolutionScaleFactor;
paramPETsimu.image_size.voxelSize_mm = PETsimu.image_size.voxelSize_mm ./ resolutionScaleFactor;
PETsimu.Revise(paramPETsimu);
%% INIT CLASS GPET FOR RECONSTRUCTION
PETrecon.scanner = 'mMR';
PETrecon.method =  'otf_siddon_gpu';
PETrecon.random_algorithm = 'from_ML_singles_matlab';
PETrecon.PSF.type = 'shift-invar';
PETrecon.PSF.Width = 2.5; %mm
PETrecon.nSubsets = 1;
PETrecon.sinogram_size.span = 11;    % Span 0 is multi slice 2d.
PETrecon = classGpet(PETrecon);
%% LOAD THE PHANTOM
load(bigBrainPhantomFilename)
fdgPhantom = phantom_atlas_density_cls.phantom;
fdgPhantomSmoothed = phantom_atlas_density_cls.phantom_smoothed;
fdgPhantomGuidedFilter = phantom_atlas_density_cls.phantom_guided_filter;
mriPhantom = phantom_atlas_density_cls.mr;
ctPhantom = phantom_atlas_density_cls.ct;
muMapPhantom = phantom_atlas_density_cls.umap;
classifiedTissuePhantom = phantom_atlas_density_cls.cls;
voxelSizeBigBrain_mm = [0.4, 0.4, 0.4];
% Create grey matter and white matter masks:
% background = 0; csf = 1; % this what its said in the txt files, but in
% fact the background is 1 and the csf is not seen
background = 1; cortical_gray = 2; white = 3; cerebellum = 4;
layer1_of_cortex = 5; sub_cortical_gray = 6; pineal_gland = 7; cerebellum_brainstem_gray = 8; cerebellum_brainstem_white = 9;
tissue_labels_value = 2:9; % I don't use the background because it shouldnt have any activity.
% Convert layer1 into gray matter:
classifiedTissuePhantom(classifiedTissuePhantom == layer1_of_cortex) = cortical_gray;
% Create masks:
maskGreyMatter = (classifiedTissuePhantom == cortical_gray) | (classifiedTissuePhantom == sub_cortical_gray) | (classifiedTissuePhantom == cerebellum_brainstem_gray);
maskWhiteMatter = (classifiedTissuePhantom == white) | (classifiedTissuePhantom == cerebellum_brainstem_white);
maskCorticalGrayMatter = (classifiedTissuePhantom == cortical_gray);
maskSubCorticalGrayMatter = (classifiedTissuePhantom == sub_cortical_gray);
%% CREATE DIFFERENT RESOLUTION VERSIONS
% The projector does not work for 0.4 mm.
% We create one for 1 mm and one for the mMR resolution.
origin_mm = [0 0 0];
% Images centred in the origin:
worldLimitsBigBrain = [origin_mm-voxelSizeBigBrain_mm.*size(fdgPhantom)/2; origin_mm+voxelSizeBigBrain_mm.*size(fdgPhantom)/2]';
worldLimitsSimu = [origin_mm-PETsimu.image_size.voxelSize_mm.*PETsimu.image_size.matrixSize/2; origin_mm+PETsimu.image_size.voxelSize_mm.*PETsimu.image_size.matrixSize/2]';
worldLimitsRecon = [origin_mm-PETrecon.image_size.voxelSize_mm.*PETrecon.image_size.matrixSize/2; origin_mm+PETrecon.image_size.voxelSize_mm.*PETrecon.image_size.matrixSize/2]';

imrefBigBrain = imref3d(size(fdgPhantom), worldLimitsBigBrain(2,:), worldLimitsBigBrain(1,:), worldLimitsBigBrain(3,:));
imrefSimu = imref3d(PETsimu.image_size.matrixSize, worldLimitsSimu(2,:), worldLimitsSimu(1,:), worldLimitsSimu(3,:));
imrefRecon = imref3d(PETrecon.image_size.matrixSize, worldLimitsRecon(2,:), worldLimitsRecon(1,:), worldLimitsRecon(3,:));

% Resample images for simulation:
[fdgPhantomSimu, refResampledImageSimu] = ImageResample(fdgPhantom, imrefBigBrain, imrefSimu, 'linear');
[mriPhantomSimu, refResampledImageSimu] = ImageResample(mriPhantom, imrefBigBrain, imrefSimu, 'linear');
[muMapPhantomSimu, refResampledImageSimu] = ImageResample(muMapPhantom, imrefBigBrain, imrefSimu, 'linear'); 
[classifiedTissuePhantomSimu, refResampledImageSimu] = ImageResample(classifiedTissuePhantom, imrefBigBrain, imrefSimu, 'nearest');
[maskGreyMatterPhantomSimu, refResampledImageSimu] = ImageResample(maskGreyMatter, imrefBigBrain, imrefSimu, 'nearest');
[maskWhiteMatterPhantomSimu, refResampledImageSimu] = ImageResample(maskWhiteMatter, imrefBigBrain, imrefSimu, 'nearest');
[maskCorticalGrayMatterPhantomSimu, refResampledImageSimu] = ImageResample(maskCorticalGrayMatter, imrefBigBrain, imrefSimu, 'nearest');
[maskSubCorticalGrayMatterPhantomSimu, refResampledImageSimu] = ImageResample(maskSubCorticalGrayMatter, imrefBigBrain, imrefSimu, 'nearest');
% Save them in nifti:
niftiwrite(fdgPhantomSimu, [phantomOutputPath 'Phantom_pet'], 'Compressed', 1);
info = niftiinfo([phantomOutputPath 'Phantom_pet']);
info.PixelDimensions = PETsimu.image_size.voxelSize_mm;
info.Datatype = 'single';
niftiwriteresorted(fdgPhantomSimu, [phantomOutputPath 'Phantom_hr_pet'], info, 1);
niftiwriteresorted(mriPhantomSimu, [phantomOutputPath 'Phantom_hr_t1'], info, 1);
niftiwriteresorted(muMapPhantomSimu, [phantomOutputPath 'Phantom_hr_umap'], info, 1);
info.Datatype = 'uint8';
niftiwriteresorted(classifiedTissuePhantomSimu, [phantomOutputPath 'Phantom_hr_tissues'], info, 1);  
niftiwriteresorted(uint8(maskGreyMatterPhantomSimu*255), [phantomOutputPath 'Phantom_hr_grey_matter'], info, 1);
niftiwriteresorted(uint8(maskWhiteMatterPhantomSimu*255), [phantomOutputPath 'Phantom_hr_white_matter'], info, 1);
niftiwriteresorted(uint8(maskCorticalGrayMatterPhantomSimu*255), [phantomOutputPath 'Phantom_hr_cortical_grey_matter'], info, 1);
niftiwriteresorted(uint8(maskSubCorticalGrayMatterPhantomSimu*255), [phantomOutputPath 'Phantom_hr_subcortical_grey_matter'], info, 1);

% Resample images for reconstruction:
[fdgPhantomRecon, refResampledImageRecon] = ImageResample(fdgPhantom, imrefBigBrain, imrefRecon, 'linear');
[mriPhantomRecon, refResampledImageRecon] = ImageResample(mriPhantom, imrefBigBrain, imrefRecon, 'linear');
[muMapPhantomRecon, refResampledImageRecon] = ImageResample(muMapPhantom, imrefBigBrain, imrefRecon, 'linear');
[classifiedTissuePhantomRecon, refResampledImageRecon] = ImageResample(classifiedTissuePhantom, imrefBigBrain, imrefRecon, 'nearest');
[maskGreyMatterPhantomRecon, refResampledImageRecon] = ImageResample(maskGreyMatter, imrefBigBrain, imrefRecon, 'nearest');
[maskWhiteMatterPhantomRecon, refResampledImageRecon] = ImageResample(maskWhiteMatter, imrefBigBrain, imrefRecon, 'nearest');
[maskCorticalGrayMatterPhantomRecon, refResampledImageRecon] = ImageResample(maskCorticalGrayMatter, imrefBigBrain, imrefRecon, 'nearest');
[maskSubCorticalGrayMatterPhantomRecon, refResampledImageRecon] = ImageResample(maskSubCorticalGrayMatter, imrefBigBrain, imrefRecon, 'nearest');
% Save them in nifti:
niftiwrite(fdgPhantomRecon, [phantomOutputPath 'Phantom_pet'], 'Compressed', 1);
info = niftiinfo([phantomOutputPath 'Phantom_pet']);
info.PixelDimensions = PETrecon.image_size.voxelSize_mm;
info.Datatype = 'single';
niftiwriteresorted(fdgPhantomRecon, [phantomOutputPath 'Phantom_pet'], info, 1);
niftiwriteresorted(mriPhantomRecon, [phantomOutputPath 'Phantom_t1'], info, 1);
niftiwriteresorted(muMapPhantomRecon, [phantomOutputPath 'Phantom_umap'], info, 1);
info.Datatype = 'uint8';
niftiwriteresorted(classifiedTissuePhantomRecon, [phantomOutputPath 'Phantom_tissues'], info, 1);  
niftiwriteresorted(uint8(maskGreyMatterPhantomRecon*255), [phantomOutputPath 'Phantom_grey_matter'], info, 1);
niftiwriteresorted(uint8(maskWhiteMatterPhantomRecon*255), [phantomOutputPath 'Phantom_white_matter'], info, 1);
niftiwriteresorted(uint8(maskCorticalGrayMatterPhantomRecon*255), [phantomOutputPath 'Phantom_cortical_grey_matter'], info, 1);
niftiwriteresorted(uint8(maskSubCorticalGrayMatterPhantomRecon*255), [phantomOutputPath 'Phantom_subcortical_grey_matter'], info, 1);

%% SIMULATION AND RECONSTRUCTION
% I simulate different number of detected events for different dose levels.
% The number of simulations per dose level changes to get to 100% of the dose.
numIterations = 60;
% Update nift info structure for the reconstruction image size:
info.PixelDimensions = PETrecon.image_size.voxelSize_mm;
info.ImageSize = PETrecon.image_size.matrixSize;
info.Datatype = 'single';
% Coutns to simulate:
fovFactor = 0.8; % The simulation does not account for the activity in the rest of the head.
counts100perc = 469313098*fovFactor;
countsPorcentaje = [100,50,25,10,5,1];
countsArray = round(counts100perc.*countsPorcentaje./100);
% The noralization is the same for all thec ases:
ncf = PETsimu.NCF; 
nf = ncf; 
nf(nf~=0) = 1./ nf(nf~=0); 
for count = 1:size(countsArray,2)
    outputPathThisLevel = [outputPath '/' num2str(countsPorcentaje(count)) '/'];
    if ~isdir(outputPathThisLevel)
        mkdir(outputPathThisLevel)
    end    
    numSimulationsThisDose = round(100/countsPorcentaje(count));

    % Counts to simulate:
    counts = countsArray(count); % Counts in the scaled ground truth.
    randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
    scatterFraction = 0.25; %efectos de la radiacion dispersa
    truesFraction = 1 - randomsFraction - scatterFraction;

    % Geometrical projection:
    y = PETsimu.P(fdgPhantomSimu); % for any other span

    % Multiplicative correction factors:
    acf= PETsimu.ACF(muMapPhantomSimu, refResampledImageSimu);
    %% Convert into factors:
    af = acf; 
    af(af~=0) = 1./ af(af~=0);
    anf = af.*nf;
    % Introduce poission noise:
    y = y.*anf;
    scale_factor = counts*truesFraction/sum(y(:));
    y = y.*scale_factor;

    % Additive factors:
    counts_randoms = counts*randomsFraction;
    r_withNorm = PETsimu.R(counts_randoms, ncf); % I pass the total coutns required and normalization factors).
          

    counts_scatter = counts*scatterFraction;
    s_withoutNorm = PETsimu.S(y);
    s_withNorm = s_withoutNorm.*nf;
    scale_factor_scatter = counts_scatter/sum(s_withNorm(:));
    s_withNorm = s_withNorm .* scale_factor_scatter;

    for i = 1 : numSimulationsThisDose
        % Poisson distribution:
        r_poisson = poissrnd(r_withNorm);
        y_poisson = poissrnd(y);
        s_poisson = poissrnd(s_withNorm);
        % Add randoms and scatter@ and poisson noise
        simulatedSinogram = y_poisson + s_poisson + r_poisson;
    
    
        % RECONSTRUCT the sinogram
        sensImage = PETrecon.Sensitivity(anf);
        recon = PETrecon.ones();
        additive = s_withNorm+r_withNorm;
        noisyDataSet3d = PETrecon.OPOSEM(simulatedSinogram, anf, additive, sensImage,recon, ceil(numIterations/PETrecon.nSubsets));
        %noisyDataSet2d{n} = permute(noisyDataSet2d{n}, [2 1 3]);
        %noisyDataSet2d{n} = noisyDataSet2d{n}(:,:,end:-1:1);
        niftiwriteresorted(noisyDataSet3d, [outputPathThisLevel sprintf('noisyDataSet%dperc_%d.nii',countsPorcentaje(count), i)], info, 1);
    end
end