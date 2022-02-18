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

%% BRAIN WEB IMAGES
% se cargan y se generan los fantomas

brainWebPath = 'C:\Users\Encargado\Milagros\BrainWEB\'
imgDir = dir ([brainWebPath])


for i = 3:length(imgDir)
    n = i-2;
    [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, refImage] = createPETPhantomFromBrainweb(strcat(brainWebPath,imgDir(i).name), [344 344 127], [2.08625 2.08625 2.03125]);

    pet_rescaled_all_images(:,:,:,n) = pet_rescaled;
    mumap_rescaled_rescaled_all_images(:,:,:,n) = mumap_rescaled;
    t1_rescaled_all_images(:,:,:,n) = t1_rescaled;
    t2_rescaled_all_images(:,:,:,n) = t2_rescaled;
    classified_tissue_rescaled_all_images(:,:,:,n) = classified_tissue_rescaled;
    refImage_all_images(:,:,:,n) = refImage;
end



%% PHANTOM TO USE

% Change the image sie, to the one of the phantom:
PET.init_image_properties(refImage_all_images(:,:,:,1));


%% SET APIRL FOR 2D
% Change sinogram size:
%param.sinogram_size.nRadialBins = 343;  % Leave the standard for mmr.
%param.sinogram_size.nAnglesBins = 252;
% aca confugaramos para las caracteristicas de 2D
param.nSubsets = 1;
param.sinogram_size.span = -1;    % Span 0 is multi slice 2d.
param.sinogram_size.nRings = 1; % for span 1, 1 ring.
param.image_size.matrixSize = [refImage_all_images(:,:,:,1).ImageSize(1:2) 1];
PET.Revise(param);
%% NOISE LEVEL
countsInGreyMatterVoxels = 8;
% Grey matter value:
greyMatterVoxelValues = max(pet_rescaled_all_images(:));
% Scale factor:
scaleFactor = countsInGreyMatterVoxels./greyMatterVoxelValues;
%% GENERATE SIMULATED DATA SET 1
% Recontruimos a nivel imagen 
% Naive approach in iamge space.

% Para cada sujeto ... 


for n = 1 : size(pet_rescaled_all_images,4)
    indicesSlices = find(sum(sum(pet_rescaled_all_images(:,:,:,n)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        groundTruth(:,:,i,n) = pet_rescaled_all_images(:,:,indicesSlices(i)); 
        groundTruthScaled(:,:,i,n) = groundTruth(:,:,i,n).*scaleFactor; % escalamos por el factor 
        noisyDataSet1(:,:,i,n) = poissrnd(groundTruthScaled(:,:,i,n)); % generamos ruido poisson sobre la imagen
    end
end


%% Show the full dataset 
% Se abriran 20 ventanas (1 por cada sujeto)

for n = 1 : size(pet_rescaled_all_images,4)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled(:,:,:,n)))); % use the same scale for the simulated data.
    figure;
    
    for i = 1 : size(groundTruthScaled,3,n)
        subplot(1,2,1);
        imshow(groundTruthScaled(:,:,i,n),[0 scaleForVisualization]);
        subplot(1,2,2);
        imshow(noisyDataSet1(:,:,i,n),[0 scaleForVisualization]);
        pause(0.1);
    end

end

%% Mostrar un slice de cada sujeto (con/sin ruido) 
num_slice = 60

% Mismos slices de diferentes sujetos - Phantom orig
figure;
for n = 1: size(pet_rescaled_all_images,4)
    subplot(4,5,n), imshow(pet_rescaled_all_images(:,:,num_slice,n),[]) 
end

% Mismos slices de diferentes sujetos -- RUIDO
figure;
for n = 1: size(pet_rescaled_all_images,4)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled(:,:,:,n))));
    subplot(4,5,n), imshow(noisyDataSet1(:,:,num_slice,n),[0 scaleForVisualization]) 
end


%% Guardar archivos y variables ..

save('2d-PhantomFromBrainWeb.mat') %

save('2d-noisyDataSet1FromBrainWeb-AllSubject.mat','noisyDataSet1','-v7.3')

save('2d-ClassifiedTissueAllSubject.mat', 'classified_tissue_rescaled_all_images', '-v7.3')
save('2d-groundTruthDataSet1FromBrainWeb-AllSubject.mat', 'groundTruth', '-v7.3')
save('2d-MuMapAllSubject.mat', 'mumap_rescaled_rescaled_all_images', '-v7.3')
save('2d-namesSubjectsBrainWeb.mat', 'namesSubjects', '-v7.3')
save('2d-PetRescaledAllSubject.mat', 'pet_rescaled_all_images', '-v7.3')
save('2d-RefImageAllSubject.mat', 'refImage_all_images', '-v7.3')
save('2d-T1AllSubject.mat', 't1_rescaled_all_images', '-v7.3')
save('2d-T2AllSubject.mat', 't2_rescaled_all_images', '-v7.3')

%% Leer archivos DATA SET 1
load('2d-PhantomFromBrainWeb.mat')

load('2d-ClassifiedTissueAllSubject.mat')
load('2d-groundTruthDataSet1FromBrainWeb-AllSubject.mat')
load('2d-MuMapAllSubject.mat')
load('2d-namesSubjectsBrainWeb.mat')
load('2d-PetRescaledAllSubject.mat')
load('2d-noisyDataSet1FromBrainWeb-AllSubject.mat')
load('2d-RefImageAllSubject.mat')
load('2d-T1AllSubject.mat')


%% GENERATE SIMULATED DATA SET 2
scaleAdjustment = 50; % -> 1.6e6
% Recontruimos a nivel sinograma
% 2d simulation for each slice :

for n = 1: 1%size(pet_rescaled_all_images,4)
    for i = 1 : numel(indicesSlices)
        groundTruth(:,:,i,n) = pet_rescaled_all_images(:,:,indicesSlices(i),n); % The same as before
        groundTruthScaled(:,:,i,n) = groundTruth(:,:,i,n).*scaleFactor;
        attenuationMap(:,:,i,n) = mumap_rescaled_rescaled_all_images(:,:,indicesSlices(i),n); % para la recontruccion
   
        % Counts to simulate:
        counts = sum(sum(groundTruthScaled(:,:,i,n))) * scaleAdjustment; % Counts in the scaled ground truth.
        randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
        scatterFraction = 0.25; %efectos de la radiacion dispersa
        truesFraction = 1 - randomsFraction - scatterFraction;

        % Geometrical projection:
        y = PET.P(groundTruthScaled(:,:,i,n) ); % for any other span

        % Multiplicative correction factors:
        acf= PET.ACF(attenuationMap(:,:,:,n), refImage_all_images(:,:,:,n));
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
        noisyDataSet2(:,:,i,n) = PET.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(60/PET.nSubsets));
    end
end
%%
for n = 1 : size(pet_rescaled_all_images,4)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled(:,:,:,n)))); % use the same scale for the simulated data.
    figure;
    
    for i = 1 : size(groundTruthScaled,3,n)
        subplot(1,2,1);
        imshow(groundTruthScaled(:,:,i,n),[0 scaleForVisualization]);
        subplot(1,2,2);
        imshow(noisyDataSet2(:,:,i,n),[0 scaleForVisualization]);
        pause(0.1);
    end

end

%% Guardar DATASET 2

save('2d-noisyDataSet2FromBrainWeb-AllSubject.mat','noisyDataSet2','-v7.3')




