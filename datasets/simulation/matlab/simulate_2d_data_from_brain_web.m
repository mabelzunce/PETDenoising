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
apirlPath = 'C:\Users\ecyt\Desktop\Milagros\apirl-code\';
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

brainWebPath = 'C:\Users\ecyt\Desktop\Milagros\BrainWEB\'
imgDir = dir ([brainWebPath])


for i = 3:length(imgDir)
    n = i-2;
    [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, refImage] = createPETPhantomFromBrainweb(strcat(brainWebPath,imgDir(i).name), [344 344 127], [2.08625 2.08625 2.03125]);

    pet_rescaled_all_images{n} = pet_rescaled;
    mumap_rescaled_rescaled_all_images{n} = mumap_rescaled;
    t1_rescaled_all_images{n} = t1_rescaled;
    t2_rescaled_all_images{n} = t2_rescaled;
    classified_tissue_rescaled_all_images{n} = classified_tissue_rescaled;
    refImage_all_images{n} = refImage;
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
param.sinogram_size.span = -1;    % Span 0 is multi slice 2d.
param.sinogram_size.nRings = 1; % for span 1, 1 ring.
param.image_size.matrixSize = [refImage_all_images{1}.ImageSize(1:2) 1];
PET.Revise(param);
%% NOISE LEVEL
countsInGreyMatterVoxels = 8;
% Grey matter value:
% greyMatterVoxelValues = max(pet_rescaled_all_images{1});
greyMatterVoxelValues = 128;
% Scale factor:
scaleFactor = countsInGreyMatterVoxels./greyMatterVoxelValues;
%% GENERATE SIMULATED DATA SET 1
% Recontruimos a nivel imagen 
% Naive approach in iamge space.

% Para cada sujeto ... 


for n = 1 : size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        groundTruth{n}(:,:,i) = pet_rescaled_all_images{n}(:,:,indicesSlices(i)); 
        groundTruthScaled{n}(:,:,i) = groundTruth{n}(:,:,i).*scaleFactor; % escalamos por el factor 
        noisyDataSet1{n}(:,:,i) = poissrnd(groundTruthScaled{n}(:,:,i)); % generamos ruido poisson sobre la imagen
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
num_slice = 43

% Mismos slices de diferentes sujetos - Phantom orig
figure;
for n = 1: size(pet_rescaled_all_images,2)
    subplot(4,5,n), imshow(pet_rescaled_all_images{n}(:,:,num_slice),[]) 
end

% Mismos slices de diferentes sujetos -- RUIDO
figure;
for n = 1: size(pet_rescaled_all_images,2)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled{n})));
    subplot(4,5,n), imshow(noisyDataSet1{n}(:,:,num_slice),[0 scaleForVisualization]) 
end

%% Analisis de dataset COMPLETO

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet1{n}(:,:,i) = (noisyDataSet1{n}(:,:,i)) .* greymask;
    end
end

for n = 1: size(pet_rescaled_all_images,2)
    mask_noisyDataSet1{n}(mask_noisyDataSet1{n}==0) =[];
end

meanValue_noisyDataSet1 = mean(cell2mat(mask_noisyDataSet1))
stdValue_noisyDataSet1 = std(cell2mat(mask_noisyDataSet1))

factorMeanStd_dataSet1 = meanValue_noisyDataSet1/stdValue_noisyDataSet1

%% Guardar archivos y variables ..

save('2d-PhantomFromBrainWeb.mat') %

save('2d-noisyDataSet1v2FromBrainWeb-AllSubject.mat','noisyDataSet1','-v7.3')

save('2d-ClassifiedTissueAllSubject.mat', 'classified_tissue_rescaled_all_images', '-v7.3')
save('2d-groundTruthDataSet1FromBrainWeb-AllSubject.mat', 'groundTruth', '-v7.3')
save('2d-MuMapAllSubject.mat', 'mumap_rescaled_rescaled_all_images', '-v7.3')
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
load('2d-noisyDataSet1v2FromBrainWeb-AllSubject.mat')
load('2d-RefImageAllSubject.mat')
load('2d-T1AllSubject.mat')


%% GENERATE SIMULATED DATA SET 2

% Calculo de normalizacion/escala
scaleAdjustment = 21; % -> 1.6e6

for n = 1: 1%size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 43
        
        groundTruth{n}(:,:,i) = pet_rescaled_all_images{n}(:,:,indicesSlices(i)); % The same as before
        groundTruthScaled{n}(:,:,i) = groundTruth{n}(:,:,i).*scaleFactor;
        attenuationMap{n}(:,:,i) = mumap_rescaled_rescaled_all_images{n}(:,:,indicesSlices(i)); % para la recontruccion
   
        % Counts to simulate:
        counts = sum(sum(groundTruthScaled{n}(:,:,i))) * scaleAdjustment; % Counts in the scaled ground truth.
        randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
        scatterFraction = 0.25; %efectos de la radiacion dispersa
        truesFraction = 1 - randomsFraction - scatterFraction;

        % Geometrical projection:
        y = PET.P(groundTruthScaled{n}(:,:,i)); % for any other span

        % Multiplicative correction factors:
        acf= PET.ACF(attenuationMap{n}(:,:,:), refImage_all_images{n}(:,:,:));
        % Convert into factors:
        af = acf;
        af(af~=0) = 1./ af(af~=0);
        % Introduce poission noise:
        y = y.*af;
        scale_factor = counts*truesFraction/sum(y(:));
        y_poisson = y.*scale_factor;

        % Additive factors:
        r = PET.R(counts*randomsFraction); 
        % Poisson distribution:
        r = r;

        counts_scatter = counts*scatterFraction;
        s_withoutNorm = PET.S(y);
        scale_factor_scatter = counts_scatter/sum(s_withoutNorm(:));
        s_withoutNorm = s_withoutNorm .* scale_factor_scatter;
        % noise for the scatter:
        s = s_withoutNorm;
        % Add randoms and scatter@ and poisson noise
        simulatedSinogram = y_poisson + s + r;


        % RECONSTRUCT the sinogram
        sensImage = PET.Sensitivity(af);
        recon = PET.ones();
        referenceSlice = PET.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(60/PET.nSubsets));
    end
end
%% Normalización DATA SET 2

greymask_referenceSlice =(pet_rescaled_all_images{1}(:,:,indicesSlices(43))==128);
mask_referenceSlice = referenceSlice .* greymask_referenceSlice;

figure;
subplot(2,2,1), imshow(pet_rescaled_all_images{1}(:,:,indicesSlices(43)),[])
subplot(2,2,2), imshow(referenceSlice,[])
subplot(2,2,3), imshow(greymask_referenceSlice,[])
subplot(2,2,4), imshow(mask_referenceSlice)

mask_referenceSlice(mask_referenceSlice==0) = []
meanValue = mean(mask_referenceSlice)

norm = (greyMatterVoxelValues.*scaleFactor)/meanValue

%% 

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        
        groundTruth{n}(:,:,i) = pet_rescaled_all_images{n}(:,:,indicesSlices(i)); % The same as before
        groundTruthScaled{n}(:,:,i) = groundTruth{n}(:,:,i).*scaleFactor;
        attenuationMap{n}(:,:,i) = mumap_rescaled_rescaled_all_images{n}(:,:,indicesSlices(i)); % para la recontruccion
   
        % Counts to simulate:
        counts = sum(sum(groundTruthScaled{n}(:,:,i))) * scaleAdjustment; % Counts in the scaled ground truth.
        randomsFraction = 0.1;  %eventos que coinciden en tiempo pero no son de la linea trazada
        scatterFraction = 0.25; %efectos de la radiacion dispersa
        truesFraction = 1 - randomsFraction - scatterFraction;

        % Geometrical projection:
        y = PET.P(groundTruthScaled{n}(:,:,i)); % for any other span

        % Multiplicative correction factors:
        acf= PET.ACF(attenuationMap{n}(:,:,:), refImage_all_images{n}(:,:,:));
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
        noisyDataSet2{n}(:,:,i) = PET.OPOSEM(simulatedSinogram,s+r, sensImage,recon, ceil(60/PET.nSubsets));
        noisyDataSet2{n}(:,:,i) = noisyDataSet2{n}(:,:,i).*norm;
    end
end


%% Analisis Slice 43 , DATA SET 1 y DATA SET 2

greymask_referenceSlice =(pet_rescaled_all_images{1}(:,:,indicesSlices(43))==128);
mask_noisyDataSet1_slice43 = noisyDataSet1{1}(:,:,43) .* greymask_referenceSlice;

% figure;
% subplot(2,2,1), imshow(pet_rescaled_all_images{1}(:,:,indicesSlices(43)),[])
% subplot(2,2,2), imshow(noisyDataSet1{1}(:,:,43),[0 scaleForVisualization])
% subplot(2,2,3), imshow(greymask_referenceSlice,[])
% subplot(2,2,4), imshow(mask_noisyDataSet1_slice43)

mask_noisyDataSet1_slice43(mask_noisyDataSet1_slice43==0) = [];
meanValue_noisyDataSet1_slice43 = mean(mask_noisyDataSet1_slice43)
stdValue_noisyDataSet1_slice43 = std(mask_noisyDataSet1_slice43)

factorMeanStd_slice43NoisyDataSet1 = meanValue_noisyDataSet1_slice43/stdValue_noisyDataSet1_slice43;

greymask_referenceSlice =(pet_rescaled_all_images{1}(:,:,indicesSlices(43))==128);
mask_noisyDataSet2_slice43 = noisyDataSet2{1}(:,:,43) .* greymask_referenceSlice;

% figure;
% subplot(2,2,1), imshow(pet_rescaled_all_images{1}(:,:,indicesSlices(43)),[])
% subplot(2,2,2), imshow(noisyDataSet2{1}(:,:,43),[0 scaleForVisualization])
% subplot(2,2,3), imshow(greymask_referenceSlice,[])
% subplot(2,2,4), imshow(mask_noisyDataSet2_slice43)

mask_noisyDataSet2_slice43(mask_noisyDataSet2_slice43==0) = [];
meanValue_noisyDataSet2_slice43 = mean(mask_noisyDataSet2_slice43)
stdValue_noisyDataSet2_slice43 = std(mask_noisyDataSet2_slice43)

factorMeanStd_slice43NoisyDataSet2 = meanValue_noisyDataSet2_slice43/stdValue_noisyDataSet2_slice43;

figure;
subplot(1,3,1), imshow(pet_rescaled_all_images{1}(:,:,indicesSlices(43)),[])
subplot(1,3,2), imshow(noisyDataSet1{1}(:,:,43),[0 scaleForVisualization])
subplot(1,3,3), imshow(noisyDataSet2{1}(:,:,43),[0 scaleForVisualization])

%% %% Analisis de cada uno de los SLICES Sujeto 1

clear var mask_noisyDataSet1
clear var mask_noisyDataSet2

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet1{n}(:,:,i) = (noisyDataSet1{n}(:,:,i)) .* greymask;
    end
end

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet2{n}(:,:,i) = (noisyDataSet2{n}(:,:,i)) .* greymask;
    end
end


% data set 1 

for n = 1:1
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0)));
    for i = 1 : numel(indicesSlices)
        array = mask_noisyDataSet1{n}(:,:,i);
        array(array == 0) = [];
        stdSliceDataSet1(i) = std(array);
    end
    arraySlice = mask_noisyDataSet1{n};
    arraySlice(arraySlice == 0) = [];
    meanSliceDataSet1 = mean(arraySlice);
end



% data set 2
for n = 1: 1%size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet2{n}(:,:,i) = (noisyDataSet2{n}(:,:,i)) .* greymask;
    end
end


for n = 1:1
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0)));
    for i = 1 : numel(indicesSlices)
        array = mask_noisyDataSet2{n}(:,:,i);
        array(array == 0) = [];
        stdSliceDataSet2(i) = std(array);
    end
    
    arraySlice = mask_noisyDataSet2{n};
    arraySlice(arraySlice == 0) = [];
    meanSliceDataSet2 = mean(arraySlice);
end


figure,
subplot(1,2,1),scatter([0:max(stdSliceDataSet1)/85:max(stdSliceDataSet1)],stdSliceDataSet1)
hold on
plot([0, max(stdSliceDataSet1)], [meanSliceDataSet1, meanSliceDataSet1])

subplot(1,2,2),scatter([0:max(stdSliceDataSet2)/85:max(stdSliceDataSet2)],stdSliceDataSet2)
hold on
plot([0, max(stdSliceDataSet2)], [meanSliceDataSet2, meanSliceDataSet2])

%% Mostrar un slice de cada sujeto (con/sin ruido) 
num_slice = 60

% Mismos slices de diferentes sujetos - Phantom orig
figure;
for n = 1: 1%size(pet_rescaled_all_images,2)
    subplot(4,5,n), imshow(pet_rescaled_all_images{n}(:,:,num_slice),[]) 
end

% Mismos slices de diferentes sujetos -- RUIDO
figure;
for n = 1: 1%size(pet_rescaled_all_images,2)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled{n})));
    subplot(4,5,n), imshow(noisyDataSet2{n}(:,:,num_slice),[0 scaleForVisualization]) 
end
figure, 
scaleForVisualization = 1.2*max(max(max(groundTruthScaled{1})));
subplot(1,2,1), imshow(noisyDataSet1{1}(:,:,43),[0 scaleForVisualization])
subplot(1,2,2), imshow(noisyDataSet2{1}(:,:,43),[0 scaleForVisualization])
%% Guardar DATASET 2

save('2d-noisyDataSet2FromBrainWebAllSubject.mat','noisyDataSet2','-v7.3')

%%
load('2d-noisyDataSet1FromBrainWeb-AllSubject.mat')
load('2d-noisyDataSet2FromBrainWebAllSubject.mat')
load('2d-PetRescaledAllSubject.mat')
load('2d-PhantomFromBrainWeb.mat')
%% Mostrar un slice de cada sujeto (con/sin ruido) 
% DATA SET 1 Y DATA SET 2
num_slice = 60

% Mismos slices de diferentes sujetos - Phantom orig
figure;
for n = 1: size(pet_rescaled_all_images,2)
    subplot(4,5,n), imshow(pet_rescaled_all_images{n}(:,:,num_slice),[]) 
end

% Mismos slices de diferentes sujetos -- RUIDO
figure;
for n = 1: size(pet_rescaled_all_images,2)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled{n})));
    subplot(4,5,n), imshow(noisyDataSet1{n}(:,:,num_slice),[0 scaleForVisualization]) 
end

% Mismos slices de diferentes sujetos -- RUIDO
figure;
for n = 1: size(pet_rescaled_all_images,2)
    scaleForVisualization = 1.2*max(max(max(groundTruthScaled{n})));
    subplot(4,5,n), imshow(noisyDataSet2{n}(:,:,num_slice),[0 scaleForVisualization]) 
end

%% Analisis DATA SET 1 y DATA SET 2

clear var mask_noisyDataSet1
clear var mask_noisyDataSet2

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet1{n}(:,:,i) = (noisyDataSet1{n}(:,:,i)) .* greymask;
    end
end

% DataSet1
for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet1{n}(:,:,i) = (noisyDataSet1{n}(:,:,i)) .* greymask;
    end
end

for n = 1: size(pet_rescaled_all_images,2)
    mask_noisyDataSet1{n}(mask_noisyDataSet1{n}==0) =[];
end

meanValue_noisyDataSet1 = mean(cell2mat(mask_noisyDataSet1))
stdValue_noisyDataSet1 = std(cell2mat(mask_noisyDataSet1))

% DataSet2

for n = 1: size(pet_rescaled_all_images,2)
    indicesSlices = find(sum(sum(pet_rescaled_all_images{n}(:,:,:)>0))); % = 86 
    for i = 1 : numel(indicesSlices)
        greymask =(pet_rescaled_all_images{n}(:,:,indicesSlices(i))==128);
        mask_noisyDataSet2{n}(:,:,i) = (noisyDataSet2{n}(:,:,i)) .* greymask;
    end
end

for n = 1: size(pet_rescaled_all_images,2)
    mask_noisyDataSet2{n}(mask_noisyDataSet2{n}==0) =[];
end

meanValue_noisyDataSet2 = mean(cell2mat(mask_noisyDataSet2))
stdValue_noisyDataSet2 = std(cell2mat(mask_noisyDataSet2))

%% MSE DATA SET 1
cantSlices = 0;
for n = 1: size(pet_rescaled_all_images,2)
    cantSlices = cantSlices + size(noisyDataSet1{n},3);
end

for n = 1: 1%size(pet_rescaled_all_images,2)
    restaCuadrado{n}(:,:,:) = ((noisyDataSet1{n}(:,:,:) - groundTruth{n}(:,:,:)).^2);
end

sumTotal = 0;

for n = 1: 1%size(pet_rescaled_all_images,2)
    array = restaCuadrado{n}(:,:,:);
    array(array == 0) = [];
    sumTotal = sumTotal + sum(array);
end

cantVoxel = 344*344*cantSlices;
mse_dataSet1 = sumTotal/cantVoxel

%% MSE DATA SET 2

for n = 1: size(pet_rescaled_all_images,2)
    restaCuadrado{n}(:,:,:) = ((noisyDataSet1{n}(:,:,:) - groundTruth{n}(:,:,:)).^2);
end

sumTotal = 0;

for n = 1: size(pet_rescaled_all_images,2)
    
    array = restaCuadrado{n}(:,:,:);
    array(array == 0) = [];
    sumTotal = sumTotal + sum(array);
end

cantVoxel = 344*344*cantSlices;
mse_dataSet1 = sumTotal/cantVoxel

%% Guardar formato nifti
for n = 1: size(pet_rescaled_all_images,2)
    if n > 1
        posicion = posicion + size(noisyDataSet1{n-1},3);
    else
        posicion = 0;
    end
    for i = 1: size(noisyDataSet1{n},3)
        array = noisyDataSet1{n}(:,:,i);
        noisyDataSet1Array(:,:,i+posicion) = array;
    end
end

for n = 1: size(pet_rescaled_all_images,2)
    if n > 1
        posicion = posicion + size(noisyDataSet1{n-1},3);
    else
        posicion = 0;
    end
    for i = 1: size(noisyDataSet2{n},3)
        array = noisyDataSet2{n}(:,:,i);
        noisyDataSet2Array(:,:,i+posicion) = array;
    end
end

for n = 1: size(groundTruth,2)
    if n > 1
        posicion = posicion + size(noisyDataSet1{n-1},3);
    else
        posicion = 0;
    end
    for i = 1: size(groundTruth{n},3)
        array = groundTruth{n}(:,:,i);
        groundTruthArray(:,:,i+posicion) = array;
    end
end

%% 

groundTruthArray = permute(groundTruthArray, [2 1 3]);
noisyDataSet1Array = permute(noisyDataSet1Array, [2 1 3]);
noisyDataSet2Array = permute(noisyDataSet2Array, [2 1 3]);

groundTruthArray = groundTruthArray(:,:,end:-1:1);
noisyDataSet1Array = noisyDataSet1Array(:,:,end:-1:1);
noisyDataSet2Array = noisyDataSet2Array(:,:,end:-1:1);

%%
% invertir eje columnas/filas
% dar vuelta en z

niftiwrite(noisyDataSet1Array,'noisyDataSet1.nii')
niftiwrite(noisyDataSet2Array,'noisyDataSet2.nii')
niftiwrite(groundTruthArray,'groundTruth.nii')