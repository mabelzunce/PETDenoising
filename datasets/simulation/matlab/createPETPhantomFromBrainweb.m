%  *********************************************************************
%  PETDenoising 
%  Function for creating phantoms from Brainweb
%  *********************************************************************

% Function that read the binary file from the brain web phantom.
% It uses the high resolution dimension:
% imageSizePhantom_pixels = [362 434 362];
% pixelSizePhantom_mm = [0.5 0.5 0.5];
% It returns the image resized to the standard PET mMR image size, and
% an attenuation map.
% Calling example: [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, refImage] = createPETPhantomFromBrainweb('subject43_crisp_v.rawb', [344 344 127], [2.08625 2.08625 2.03125]);

function [pet_rescaled, mumap_rescaled, t1_rescaled, t2_rescaled, classified_tissue_rescaled, maskGrayMatter, maskWhiteMatter, refImage] = createPETPhantomFromBrainweb(binaryFilename, imageSize_pixels, pixelSize_mm,contrast_ratio)
%% PARAMETERS
imageSizePhantom_pixels = [362 434 362];
pixelSizePhantom_mm = [0.5 0.5 0.5];
if nargin == 1
    % set the default pixel value and matrix size:
    % Size of the pixels:
    pixelSize_mm = [0.5 0.5 0.5];
    % The size in pixels:
    imageSize_pixels = [362 434 362];
    contrast_ratio = 4;
end
%% READ BINARY IMAGE
% Read image:
fid = fopen(binaryFilename, 'r');
if fid == -1
    ferror(fid);
end
phantom = fread(fid, imageSizePhantom_pixels(1)*imageSizePhantom_pixels(2)*imageSizePhantom_pixels(3), 'uint8');
phantom = reshape(phantom, imageSizePhantom_pixels);
% Then interchange rows and cols, x and y: 
phantom = permute(phantom, [2 1 3]);
phantom = phantom(end:-1:1,:,end:-1:1);
fclose(fid);
imageSizePhantom_pixels = size(phantom);
classified_tissue = phantom;%round(phantom./16);
%% PHANTOM PARAMETER

indicesCsf = phantom == 1;
maskWhiteMatter = phantom == 3;
maskGrayMatter = phantom == 2;
indicesFat = phantom == 4;
indicesMuscle = phantom == 5;
indicesMuscleSkin = phantom == 6;
indicesSkull = phantom == 7;
indicesVessels = phantom == 8;
indicesAroundFat = phantom == 9;
indicesMarrow = phantom == 11;
indicesDura = phantom == 10;
indicesBone = indicesSkull | indicesMarrow | indicesDura;
%% FIRST CREATE DIFFERENT MAPS AND THEN RESIZE TO THE SIZE WANTED
%% CREATE ATTENUATION MAP
% brainweb material id:
% 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter, 4=Fat, 5=Muscle, 6=Muscle/Skin, 7=Skull, 8=vessels, 9=around fat, 10 =dura matter, 11=bone marrow
mumap = phantom;
% Set the attenuation of bones:
mu_bone_1_cm = 0.13;
mu_bone_1_cm = 0.13;
mu_tissue_1_cm = 0.0975;
mumap(phantom >0) = mu_tissue_1_cm;
mumap(indicesBone) = mu_bone_1_cm;

%% TRANSFORM THE ATANOMY INTO PET SIGNALS
whiteMatterAct = 32;
grayMatterAct = whiteMatterAct * contrast_ratio;
skinAct = 16;
pet = phantom;
pet(maskWhiteMatter) = whiteMatterAct;
pet(maskGrayMatter) = grayMatterAct;
pet(indicesMuscleSkin) = skinAct;
pet(~maskWhiteMatter & ~maskGrayMatter & ~indicesMuscleSkin) = 0;
refImage = imref3d(imageSize_pixels, pixelSize_mm(2), pixelSize_mm(1), pixelSize_mm(3));

%% T1
t1 = phantom;
whiteMatterT1 = 154;
grayMatterT1 = 106;
skinT1 = 92;
skullT1 = 48;
marrowT1 = 180;
duraT1 = 48;
csfT2 = 48;
t1(maskWhiteMatter) = whiteMatterT1;
t1(maskGrayMatter) = grayMatterT1;
t1(indicesMuscleSkin) = skinT1;
t1(~maskWhiteMatter & ~maskGrayMatter & ~indicesMuscleSkin & ~indicesBone) = 0;
t1(indicesSkull) = skullT1;
t1(indicesMarrow) = marrowT1;
t1(indicesBone) = duraT1;
t1(indicesCsf) = csfT2;
%% T2
t2 = phantom;
whiteMatterT2 = 70;
grayMatterT2 = 100;
skinT2 = 70;
skullT2 = 100;
marrowT2 = 250;
csfT2 = 250;
duraT2 = 200;
t2(maskWhiteMatter) = whiteMatterT2;
t2(maskGrayMatter) = grayMatterT2;
t2(indicesMuscleSkin) = skinT2;
t2(~maskWhiteMatter & ~maskGrayMatter & ~indicesMuscleSkin & ~indicesBone) = 0;
t2(indicesCsf) = csfT2;
t2(indicesSkull) = skullT2;
t2(indicesMarrow) = marrowT2;
t2(indicesBone) = duraT2;

%% CONVERT TO DEFAULT SIZE
% Size of the image to cover the full fov:
sizeImage_mm = pixelSize_mm .* imageSize_pixels;
if pixelSize_mm ~= pixelSizePhantom_mm | imageSize_pixels ~= imageSizePhantom_pixels
    % Resample to the emission images space (centered axially):
    coordXphantom = (-pixelSizePhantom_mm(2)*imageSizePhantom_pixels(2)/2+pixelSizePhantom_mm(2)/2) : pixelSizePhantom_mm(2) : (pixelSizePhantom_mm(2)*imageSizePhantom_pixels(2)/2);
    coordYphantom = (-pixelSizePhantom_mm(1)*imageSizePhantom_pixels(1)/2+pixelSizePhantom_mm(1)/2) : pixelSizePhantom_mm(1) : (pixelSizePhantom_mm(1)*imageSizePhantom_pixels(1)/2);
    coordZphantom = (-pixelSizePhantom_mm(3)*imageSizePhantom_pixels(3)/2+pixelSizePhantom_mm(3)/2) : pixelSizePhantom_mm(3) : (pixelSizePhantom_mm(3)*imageSizePhantom_pixels(3)/2);
    [Xphantom, Yphantom, Zphantom] = meshgrid(coordXphantom, coordYphantom, coordZphantom);
    % Idem for the dixon attenuation map:
    coordXpet = (-pixelSize_mm(2)*imageSize_pixels(2)/2 + pixelSize_mm(2)/2) : pixelSize_mm(2) : (pixelSize_mm(2)*imageSize_pixels(2)/2);
    coordYpet = (-pixelSize_mm(1)*imageSize_pixels(1)/2 + pixelSize_mm(1)/2) : pixelSize_mm(1) : (pixelSize_mm(1)*imageSize_pixels(1)/2);
    coordZpet = (-pixelSize_mm(3)*imageSize_pixels(3)/2 + pixelSize_mm(3)/2) : pixelSize_mm(3) : (pixelSize_mm(3)*imageSize_pixels(3)/2);
    [Xpet, Ypet, Zpet] = meshgrid(coordXpet,coordYpet, coordZpet);
    % Interpolate the phantom image to the wanted coordinates:
    pet_rescaled = single(interp3(Xphantom,Yphantom,Zphantom,pet,Xpet,Ypet,Zpet)); 
    pet_rescaled(isnan(pet_rescaled)) = 0;
    mumap_rescaled = single(interp3(Xphantom,Yphantom,Zphantom,mumap,Xpet,Ypet,Zpet)); 
    mumap_rescaled(isnan(mumap_rescaled)) = 0;
    t1_rescaled = single(interp3(Xphantom,Yphantom,Zphantom,t1,Xpet,Ypet,Zpet)); 
    t1_rescaled(isnan(t1_rescaled)) = 0;
    t2_rescaled = single(interp3(Xphantom,Yphantom,Zphantom,t2,Xpet,Ypet,Zpet)); 
    t2_rescaled(isnan(t2_rescaled)) = 0;
    classified_tissue_rescaled = interp3(Xphantom,Yphantom,Zphantom,classified_tissue,Xpet,Ypet,Zpet, 'nearest'); 
    classified_tissue_rescaled(isnan(classified_tissue_rescaled)) = 0;
    maskWhiteMatter = interp3(Xphantom,Yphantom,Zphantom,uint8(maskWhiteMatter),Xpet,Ypet,Zpet, 'nearest'); 
    maskWhiteMatter(isnan(maskWhiteMatter)) = 0;
    maskGrayMatter = interp3(Xphantom,Yphantom,Zphantom,uint8(maskGrayMatter),Xpet,Ypet,Zpet, 'nearest'); 
    maskGrayMatter(isnan(maskGrayMatter)) = 0;

    % I am having problems with the first slices, zero padd them:
    pet_rescaled(:,:,1:5) = 0;
    mumap_rescaled(:,:,1:5) = 0;
    t1_rescaled(:,:,1:5) = 0;
    t2_rescaled(:,:,1:5) = 0;
    classified_tissue_rescaled(:,:,1:5) = 0;
    maskGrayMatter(:,:,1:5) = 0;
    maskWhiteMatter(:,:,1:5) = 0;
else
    pet_rescaled = single(pet);
    mumap_rescaled = single(mumap);
    t1_rescaled = single(t1);
    t2_rescaled = single(t2);
    classified_tissue_rescaled = classified_tissue;
end
