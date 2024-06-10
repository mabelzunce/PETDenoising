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

outputPath = 'D:\UNSAM\PET\BrainWebSimulations\';
if ~isdir(outputPath)
    mkdir(outputPath)
end
%% INIT CLASS GPET

PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.random_algorithm = 'from_ML_singles_matlab';
objGpet.PSF.type = 'None';
objGpet.PSF.Width = 2.5; %mm
PET = classGpet(PET);

%% BRAIN WEB IMAGES
% se cargan y se generan los fantomas

brainWebPath = 'D:\UNSAM\PET\BrainWEB\';
imgDir = dir ([brainWebPath]);
contrastRatio = [2,4,6]


for i = 3:length(imgDir)
	n = i-2;
	for j = 1:length(contrastRatio)
	    [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, maskGrayMatter, maskWhiteMatter, refImage] = createPETPhantomFromBrainweb(strcat(brainWebPath,imgDir(i).name), [344 344 127], [2.08625 2.08625 2.03125],j);

	    pet_rescaled_all_images{n} = single(pet_rescaled);
	    mumap_rescaled_rescaled_all_images{n} = single(mumap_rescaled);
	    t1_rescaled_all_images{n} =  single(t1_rescaled);
	    t2_rescaled_all_images{n} =  single(t2_rescaled);
	    classified_tissue_rescaled_all_images{n} =  uint8(classified_tissue_rescaled);
	    refImage_all_images{n} = refImage;
	    if n == 1
		niftiwrite(pet_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], 'Compressed', 1);
		info = niftiinfo([outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)]);
		info.PixelDimensions = PET.image_size.voxelSize_mm;
	    end
	    info.Datatype = 'single';
	    niftiwriteresorted(pet_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	    niftiwriteresorted(mumap_rescaled_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	    niftiwriteresorted(t1_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	    niftiwriteresorted(t2_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	    info.Datatype = 'uint8';
	    niftiwriteresorted(classified_tissue_rescaled_all_images{n}, [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);  
	    niftiwriteresorted(uint8(maskGrayMatter*255), [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	    niftiwriteresorted(uint8(maskWhiteMatter*255), [outputPath sprintf('Phantom_%d_pet_ContrastRatio__%d', n,j)], info, 1);
	end
end

%% PHANTOM TO USE

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refImage_all_images{1});


%% SET APIRL FOR 2D
% Change sinogram size:
%param.sinogram_size.nRadialBins = 343;  % Leave the standard for mmr.
%param.sinogram_size.nAnglesBins = 252;
% aca confugaramos para las caracteristicas de 2D
param.nSubsets = 1;
param.sinogram_size.span = 11;    % Span 0 is multi slice 2d.
% param.sinogram_size.nRings = 3; % for span 1, 1 ring.
%param.image_size.matrixSize = [refImage_all_images{1}.ImageSize(1:2) 1];
PET.Revise(param);
% structSizeSino3d = getSizeSino3dFromSpan(PET.sinogram_size.nRadialBins, PET.sinogram_size.nAnglesBins, PET.sinogram_size.Rings, 0, ...
%     0, PET.sinogram_size.span, PET.sinogram_size.maxRingDiffs);
%%
fovFactor = 0.8; % The simulation does not account for the activity in the rest of the head.
counts100perc = 469313098*fovFactor;
countsPorcentaje = [100 5 1 25 50];%[100,50,25,10,5,1];
countsArray = round(counts100perc.*countsPorcentaje./100);

%%

for count = 1:size(countsArray,2)
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
        y = PET.P(groundTruthScaled{n}); % for any other span

        % Multiplicative correction factors:
        acf= PET.ACF(attenuationMap{n}, refImage_all_images{n});
        % Convert into factors:
        af = acf;
        af(af~=0) = 1./ af(af~=0);
        % Introduce poission noise:
        y = y.*af;
        scale_factor = counts*truesFraction/sum(y(:));
        y = y.*scale_factor;
        y_poisson = poissrnd(y);

        % Additive factors:
        r = PET.R(counts*randomsFraction); 
        % Poisson distribution:
        r = poissrnd(r);

        counts_scatter = counts*scatterFraction;
        s_withoutNorm = PET.S(y);
        scale_factor_scatter = counts_scatter/sum(s_withoutNorm(:));
        s_withoutNorm = s_withoutNorm .* scale_factor_scatter;
        % noise for the scatter:
        s = poissrnd(s_withoutNorm);
        % Add randoms and scatter@ and poisson noise
        simulatedSinogram = y_poisson + s + r;


        % RECONSTRUCT the sinogram
        sensImage = PET.Sensitivity(af);
        recon = PET.ones();
        noisyDataSet2d{n} = PET.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(60/PET.nSubsets));
        %noisyDataSet2d{n} = permute(noisyDataSet2d{n}, [2 1 3]);
        %noisyDataSet2d{n} = noisyDataSet2d{n}(:,:,end:-1:1);
        niftiwriteresorted(noisyDataSet2d{n}, [outputPath sprintf('noisyDataSet%d_Subject%d.nii',countsPorcentaje(count),n)], info, 1);
        %niftiwrite(noisyDataSet2d{n},sprintf('noisyDataSet%d_Subject%d.nii',countsPorcentaje(count),n))
    end
end
