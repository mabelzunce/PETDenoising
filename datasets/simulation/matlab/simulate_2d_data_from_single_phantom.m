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
apirlPath = 'C:\Users\Encargado\Milagros\CodigosMatlab\apirl-code\';
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
%param.sinogram_size.nRadialBins = 343;  % Leave the standard for mmr.
%param.sinogram_size.nAnglesBins = 252;
% aca confugaramos para las caracteristicas de 2D
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
% Recontruimos a nivel imagen 
% Naive approach in iamge space.

indicesSlices = find(sum(sum(tAct>0))); % = 117 
for i = 1 : numel(indicesSlices)
    groundTruth(:,:,i) = tAct(:,:,indicesSlices(i)); 
    groundTruthScaled(:,:,i) = groundTruth(:,:,i).*scaleFactor; % escalamos por el factor 
    noisyDataSet1(:,:,i) = poissrnd(groundTruthScaled(:,:,i)); % generamos ruido poisson sobre la imagen
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
scaleAdjustment = 50; % -> 1.6e6
% Recontruimos a nivel sinograma
% 2d simulation for each slice:
for i = 1 : numel(indicesSlices)
    groundTruth(:,:,i) = tAct(:,:,indicesSlices(i)); % The same as before
    groundTruthScaled(:,:,i) = groundTruth(:,:,i).*scaleFactor;
    attenuationMap(:,:,i) = tMu(:,:,indicesSlices(i)); % para la recontruccion
   
    % Counts to simulate:
    counts = sum(sum(groundTruthScaled(:,:,i))) * scaleAdjustment; % Counts in the scaled ground truth.
    randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
    scatterFraction = 0.25; %efectos de la radiacion dispersa
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
    % Add randoms and scatter@ and poisson noise
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

%% Save workspace

filename = 'ws2d.mat';
save(filename)
%%

load('ws2d.mat')
%% MASCARAS

for i = 1 : numel(indicesSlices)
    mask_materia_gris(:,:,i) = (tAct(:,:,indicesSlices(i))==9000); 
    mask_materia_blanca(:,:,i) = (tAct(:,:,indicesSlices(i))==3400);
   
end

 noisyDataSet1_materia_gris = noisyDataSet1 .* mask_materia_gris;
 noisyDataSet1_materia_blanca = noisyDataSet1 .* mask_materia_blanca;
    
 noisyDataSet2_materia_gris = noisyDataSet2 .* mask_materia_gris;
 noisyDataSet2_materia_blanca = noisyDataSet2 .* mask_materia_blanca;
 
%% DATA SET 1

% Materia gris
figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,3,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,2);
    imshow(noisyDataSet1(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,3);
    imshow(noisyDataSet1_materia_gris(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end

% Materia blanca
figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,3,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,2);
    imshow(noisyDataSet1(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,3);
    imshow(noisyDataSet1_materia_blanca(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end

 
%% DATA SET 2

% Materia gris
figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,3,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,2);
    imshow(noisyDataSet2(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,3);
    imshow(noisyDataSet2_materia_gris(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end

% Materia blanca
figure;
for i = 1 : size(groundTruthScaled,3)
    subplot(1,3,1);
    imshow(groundTruthScaled(:,:,i),[0 scaleForVisualization]);
    subplot(1,3,2);
    imshow(noisyDataSet2(:,:,i),[0 scaleForVisualization*0.7]);
    subplot(1,3,3);
    imshow(noisyDataSet2_materia_blanca(:,:,i),[0 scaleForVisualization]);
    pause(0.1);
end

%% ANALISIS DE DATOS

noisyDataSet1_materia_gris(noisyDataSet1_materia_gris==0) = [];
noisyDataSet1_materia_blanca(noisyDataSet1_materia_blanca==0) = [];

noisyDataSet2_materia_gris(noisyDataSet2_materia_gris==0) = [];
noisyDataSet2_materia_blanca(noisyDataSet2_materia_blanca==0) = [];
    

figure, 
subplot(2,2,1), histogram(noisyDataSet1_materia_gris,22), title('SET 1 - G')
subplot(2,2,2), histogram(noisyDataSet1_materia_blanca,12), title('SET 1 - B')
subplot(2,2,3), histogram(noisyDataSet2_materia_gris,21), title('SET 2 - G')
subplot(2,2,4), histogram(noisyDataSet2_materia_blanca,22), title('SET 2 - B')








