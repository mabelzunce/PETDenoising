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

outputPath =  '../../../data/BrainWebSimulations/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
phantomOutputPath = [outputPath 'Phantoms/'];
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
%% BRAIN WEB IMAGES
% se cargan y se generan los fantomas
brainWebPath = '../../../data/BrainWebPhantoms/';
imgDir = dir ([brainWebPath]);
% First create low resolution phantoms of the same size of the
% reconstructed image:
for i = 3:length(imgDir)
    n = i-2;
    [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, maskGrayMatter, maskWhiteMatter, refImage] = createPETPhantomFromBrainweb(strcat(brainWebPath,imgDir(i).name), ...
        PETrecon.image_size.matrixSize, PETrecon.image_size.voxelSize_mm);

    pet_rescaled_all_images{n} = single(pet_rescaled);
    mumap_rescaled_rescaled_all_images{n} = single(mumap_rescaled);
    t1_rescaled_all_images{n} =  single(t1_rescaled);
    t2_rescaled_all_images{n} =  single(t2_rescaled);
    classified_tissue_rescaled_all_images{n} =  uint8(classified_tissue_rescaled);
    refImage_all_images{n} = refImage;
    if n == 1
        niftiwrite(pet_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_pet', n)], 'Compressed', 1);
        info = niftiinfo([phantomOutputPath sprintf('Phantom_%d_pet', n)]);
        info.PixelDimensions = PETrecon.image_size.voxelSize_mm;
    end
    info.Datatype = 'single';
    niftiwriteresorted(pet_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_pet', n)], info, 1);
    niftiwriteresorted(mumap_rescaled_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_mumap', n)], info, 1);
    niftiwriteresorted(t1_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_t1', n)], info, 1);
    niftiwriteresorted(t2_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_t2', n)], info, 1);
    info.Datatype = 'uint8';
    niftiwriteresorted(classified_tissue_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_tissues', n)], info, 1);  
    niftiwriteresorted(uint8(maskGrayMatter*255), [phantomOutputPath sprintf('Phantom_%d_grey_matter', n)], info, 1);
    niftiwriteresorted(uint8(maskWhiteMatter*255), [phantomOutputPath sprintf('Phantom_%d_white_matter', n)], info, 1);
end

% Now we create the high resolution phantom for the simulations:
for i = 3:length(imgDir)
    n = i-2;
    [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, maskGrayMatter, maskWhiteMatter, refImage] = createPETPhantomFromBrainweb(strcat(brainWebPath,imgDir(i).name), ...
        PETsimu.image_size.matrixSize, PETsimu.image_size.voxelSize_mm);

    pet_rescaled_all_images{n} = single(pet_rescaled);
    mumap_rescaled_rescaled_all_images{n} = single(mumap_rescaled);
    t1_rescaled_all_images{n} =  single(t1_rescaled);
    t2_rescaled_all_images{n} =  single(t2_rescaled);
    classified_tissue_rescaled_all_images{n} =  uint8(classified_tissue_rescaled);
    refImage_all_images{n} = refImage;
    if n == 1
        niftiwrite(pet_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_hr_%d_pet', n)], 'Compressed', 1);
        info = niftiinfo([phantomOutputPath sprintf('Phantom_hr_%d_pet', n)]);
        info.PixelDimensions = PETsimu.image_size.voxelSize_mm;
    end
    info.Datatype = 'single';
    niftiwriteresorted(pet_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_hr_%d_pet', n)], info, 1);
    niftiwriteresorted(mumap_rescaled_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_%d_mumap', n)], info, 1);
    niftiwriteresorted(t1_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_hr_%d_t1', n)], info, 1);
    niftiwriteresorted(t2_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_hr_%d_t2', n)], info, 1);
    info.Datatype = 'uint8';
    niftiwriteresorted(classified_tissue_rescaled_all_images{n}, [phantomOutputPath sprintf('Phantom_hr_%d_tissues', n)], info, 1);  
    niftiwriteresorted(uint8(maskGrayMatter*255), [phantomOutputPath sprintf('Phantom_hr_%d_grey_matter', n)], info, 1);
    niftiwriteresorted(uint8(maskWhiteMatter*255), [phantomOutputPath sprintf('Phantom_hr_%d_white_matter', n)], info, 1);
end
%% SIMULATION AND RECONSTRUCTION
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

for count = 1:size(countsArray,2)
    outputPathThisLevel = [outputPath '\' num2str(countsPorcentaje(count)) '\'];
    if ~isdir(outputPathThisLevel)
        mkdir(outputPathThisLevel)
    end
    for n = 1: size(pet_rescaled_all_images,2) 
        groundTruth{n} = pet_rescaled_all_images{n}; % The same as before
        groundTruthScaled{n} = groundTruth{n};
        attenuationMap{n} = mumap_rescaled_rescaled_all_images{n}; % para la recontruccion

        % Counts to simulate:
        counts = countsArray(count); % Counts in the scaled ground truth.
        randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
        scatterFraction = 0.25; %efectos de la radiacion dispersa
        truesFraction = 1 - randomsFraction - scatterFraction;

        % Geometrical projection:
        y = PETsimu.P(groundTruthScaled{n}); % for any other span

        % Multiplicative correction factors:
        acf= PETsimu.ACF(attenuationMap{n}, refImage_all_images{n});
        % Convert into factors:
        af = acf;
        af(af~=0) = 1./ af(af~=0);
        % Introduce poission noise:
        y = y.*af;
        scale_factor = counts*truesFraction/sum(y(:));
        y = y.*scale_factor;
        y_poisson = poissrnd(y);

        % Additive factors:
        r = PETsimu.R(counts*randomsFraction); 
        % Poisson distribution:
        r = poissrnd(r);

        counts_scatter = counts*scatterFraction;
        s_withoutNorm = PETsimu.S(y);
        scale_factor_scatter = counts_scatter/sum(s_withoutNorm(:));
        s_withoutNorm = s_withoutNorm .* scale_factor_scatter;
        % noise for the scatter:
        s = poissrnd(s_withoutNorm);
        % Add randoms and scatter@ and poisson noise
        simulatedSinogram = y_poisson + s + r;


        % RECONSTRUCT the sinogram
        sensImage = PETrecon.Sensitivity(af);
        recon = PETrecon.ones();
        noisyDataSet3d{n} = PETrecon.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(numIterations/PETrecon.nSubsets));
        %noisyDataSet2d{n} = permute(noisyDataSet2d{n}, [2 1 3]);
        %noisyDataSet2d{n} = noisyDataSet2d{n}(:,:,end:-1:1);
        niftiwriteresorted(noisyDataSet3d{n}, [outputPathThisLevel sprintf('noisyDataSet%d_Subject%d.nii',countsPorcentaje(count),n)], info, 1);
        %niftiwrite(noisyDataSet2d{n},sprintf('noisyDataSet%d_Subject%d.nii',countsPorcentaje(count),n))
    end
end