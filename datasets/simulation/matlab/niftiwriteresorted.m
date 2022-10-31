function niftiwriteresorted(image, outputFilename, info, compressed)
%NIFTIWRITERESORTED Writes a nifti image using matlab function but
%previously resorts it
%   Detailed explanation goes here
image = permute(image, [2 1 3]);

image = image(:,:,end:-1:1);
niftiwrite(image, outputFilename, info, 'Compressed', compressed);
end

