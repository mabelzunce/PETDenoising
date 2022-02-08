%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
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
%% INIT CLASS GPET
PET.scanner = 'cylindrical';
PET.method =  'otf_siddon_cpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.random_algorithm = 'from_ML_singles_matlab';
PET = classGpet(PET);
%% PHANTOM TO USE
load BrainMultiMaps_mMR.mat;
tAct = permute(MultiMaps_Ref.PET, [2 1 3]);
tAct = tAct(end:-1:1,:,:);
tMu = permute(MultiMaps_Ref.uMap, [2 1 3]);
tMu = tMu(end:-1:1,:,:);
pixelSize_mm = [2.08625 2.08625 2.03125];
xLimits = [-size(tAct,2)/2*pixelSize_mm(2) size(tAct,2)/2*pixelSize_mm(2)];
yLimits = [-size(tAct,1)/2*pixelSize_mm(1) size(tAct,1)/2*pixelSize_mm(1)];
zLimits = [-size(tAct,3)/2*pixelSize_mm(3) size(tAct,3)/2*pixelSize_mm(3)];
refAct = imref3d(size(tAct),xLimits,yLimits,zLimits);
refAt  = imref3d(size(tMu),xLimits,yLimits,zLimits);

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refAct);
%% SET APIRL FOR 2D
% Change sinogram size:
%param.sinogram_size.nRadialBins = 344;  % Leave the standard for mmr.
%param.sinogram_size.nAnglesBins = 252;
param.nSubsets = 1;
param.sinogram_size.span = -1;    % Span 0 is multi slice 2d.
param.sinogram_size.nRings = 1; % for span 1, 1 ring.
param.image_size.matrixSize = [refAct.ImageSize(1:2) 1];
PET.Revise(param);
%% NOISE LEVEL
countsInGreyMatterVoxels = 8;
% Grey matter value:
greyMatterVoxelValues = max(tAct(:));
% Scale factor:
scaleFactor = countsInGreyMatterVoxels./greyMatterVoxelValues;
%% GENERATE SIMULATED DATA SET 1
% Naive approach in iamge space.
indicesSlices = find(sum(sum(tAct>0)));
for i = 1 : numel(indicesSlices)
    groundTruth(:,:,i) = tAct(:,:,indicesSlices(i));
    groundTruthScaled(:,:,i) = groundTruth(:,:,i).*scaleFactor;
    noisyDataSet1(:,:,i) = poissrnd(groundTruthScaled(:,:,i));
end
% Show the full dataset:
scaleForVisualization = 1.2*max(max(max(groundTruthScaled))); % use the same scale for the simulated data.
figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,2,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,2,2);
    imshow(noisyDataSet1(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end
%% GENERATE SIMULATED DATA SET 2
scaleAdjustment = 1;
% 2d simulation for each slice:
for i = 1 : numel(indicesSlices)
    groundTruth(:,:,i) = tAct(:,:,indicesSlices(i)); % The same as before
    groundTruthScaled(:,:,i) = groundTruth(:,:,i).*scaleFactor;
    attenuationMap(:,:,i) = tMu(:,:,indicesSlices(i));
   
    % Counts to simulate:
    counts = sum(sum(groundTruthScaled(:,:,i))) * scaleAdjustment; % Counts in the scaled ground truth.
    randomsFraction = 0.1;
    scatterFraction = 0.25;
    truesFraction = 1 - randomsFraction - scatterFraction;

    % Geometrical projection:
    y = PET.P(groundTruthScaled(:,:,i) ); % for any other span

    % Multiplicative correction factors:
    acf= PET.ACF(attenuationMap, refAct);
    % Convert into factors:
    af = acf;
    af(af~=0) = 1./ af(af~=0);
    % Introduce poission noise:
    y = y.*af;
    scale_factor = counts*truesFraction/sum(y(:));
    y_poisson = poissrnd(y.*scale_factor);

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
    % Add randoms and scatter@
    simulatedSinogram = y_poisson + s + r;


    % RECONSTRUCT the sinogram
    sensImage = PET.Sensitivity(af);
    recon = PET.ones();
    noisyDataSet2(:,:,i) = PET.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(60/PET.nSubsets));
end

figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,2,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,2,2);
    imshow(noisyDataSet2(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end