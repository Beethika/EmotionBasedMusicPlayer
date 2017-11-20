
function [isSucceed] = EigenFace1(strTrainPath, strLabelFile, strTestPath)


isSucceed = 0;
if (exist('strTrainPath')==0)
   % strTrainPath = input('Enter Train Folder Name:','s');
    strTrainPath = 'Train';
end
if (exist('strLabelFile')==0)
    %strLabelFile = input('Enter Label File Name:','s');
    strLabelFile = 'LabelFile.txt';
end
if (exist('strTestPath')==0)    
   % strTestPath = input('Enter Test Folder Name:','s');
        k=input('','s')
    strTestPath = ['Test' k]
end
 
fid=fopen(strLabelFile);
imageLabel=textscan(fid,'%s %s','whitespace',',');
fclose(fid);

NeutralImages=[];
for i=1:length(imageLabel{1,1})
    if (strcmp(lower(imageLabel{1,2}{i,1}),'neutral'))
        NeutralImages=[NeutralImages,i];
    end 
end
if (length(NeutralImages)==0)
    disp('ERROR: Neutral Expression is not available in training');
    return;
end

structTestImages = dir(strTestPath);
numImage = length(imageLabel{1,1});  % Total Observations: Number of Images in training set
lenTest = length(structTestImages);

if (lenTest==0)
    disp('Error:Invalid Test Folder');
    return;
end

TrainImages='';
for i = 1:numImage
	TrainImages{i,1} = strcat(strTrainPath,'\',imageLabel{1,1}(i));
end

j=0;
for i = 3:lenTest
     if ((~structTestImages(i).isdir))
         if  (structTestImages(i).name(end-3:end)=='.jpg')
             j=j+1;
             TestImages{j,1} = [strTestPath,'\',structTestImages(i).name];
         end
     end
end
numTestImage = j; % Number of Test Images
clear ('structTestImages','fid','i','j');pack

imageSize = [280,180];          % All Images are resized into a common size

%% ################# Load Train Data & Preprocess  ########################
%% Loading training images & preparing for PCA by subtracting mean

img = zeros(imageSize(1)*imageSize(2),numImage);
for i = 1:numImage
    aa = imresize(detect_face(imresize(imread(cell2mat(TrainImages{i,1})),[375,300])),imageSize);
    img(:,i) = aa(:);
    disp(sprintf('Loading Train Image # %d',i));
end
meanImage = mean(img,2);        
                 
img = (img - meanImage*ones(1,numImage))';      % img is the input to PCA
%% ########################################################################

%% ################# Low Dimension Face Space Construction ################
[C,S,L]=princomp(img,'econ');                   % Performing PCA Here
EigenRange = [1:30];   % Defines which Eigenvalues will be selected
C = C(:,EigenRange);
%% ########################################################################


%% ############# Load Test Data and project on Face Space #################
img = zeros(imageSize(1)*imageSize(2),numTestImage);
for i = 1:numTestImage
    aa = imresize(detect_face(imresize(imread(TestImages{i,1}),[375,300])),imageSize);
    img(:,i) = aa(:);
    disp(sprintf('Loading Test Image # %d',i));
    %%
%%###########################Face & Feature Detect33333####################
img1=imread(TestImages{i,1});
%function face(img);
imgh = imresize(img1, [250 200]);
figure, imshow(imgh);
final_image = zeros(size(imgh,1), size(imgh,2));
if(size(imgh, 3) > 1)
for i1 = 1:size(imgh,1)
for j = 1:size(imgh,2)
R = imgh(i1,j,1);
G = imgh(i1,j,2);
B = imgh(i1,j,3);
if(R > 95 && G > 40 && B > 20)
v = [R,G,B];
if((max(v) - min(v)) > 15)
if(abs(R-G) > 15 && R > G && R > B)
%it is a skin
final_image(i1,j) = 1;
end
end
end
end
end
figure, imshow(final_image);
end
binaryImage=im2bw(final_image,0.6);
%figure, imshow(binaryImage);

%eroding image
se = strel('disk',11);
I2 = imerode(binaryImage,se);
figure, imshow(I2);
title('eroded image');

%dilate image
I3 = imdilate(binaryImage,se);
figure, imshow(I3);
title('dilated image');

%skeletanizing
%BW3 = bwmorph(I3,'skel',Inf);
%figure, imshow(BW3);
%title('skeletanized image');

%Filling The Holes.
binaryImage = imfill(binaryImage, 'holes');
figure, imshow(binaryImage);

binaryImage = bwareaopen(binaryImage,1890);   
figure,imshow(binaryImage);
labeledImage = bwlabel(binaryImage, 8);
blobMeasurements = regionprops(labeledImage, final_image, 'all');
numberOfPeople = size(blobMeasurements, 1);
imagesc(imgh);
title('Outlines, from bwboundaries()'); 



%axis square;
hold on;
boundaries = bwboundaries(binaryImage);
for k = 1 : numberOfPeople
thisBoundary = boundaries{k};
plot(thisBoundary(:,2), thisBoundary(:,1), 'g', 'LineWidth', 2);
end
% hold off;

imagesc(imgh);
hold on;
title('Original with bounding boxes');
%fprintf(1,'Blob # x1 x2 y1 y2\n');
%for k = 1 : numberOfPeople % Loop through all blobs.
% Find the mean of each blob. (R2008a has a better way where you can pass the original image
% directly into regionprops. The way below works for all versionsincluding earlier versions.)
thisBlobsBox = blobMeasurements(k).BoundingBox; % Get list of pixels in current blob.
x1 = thisBlobsBox(1);
y1 = thisBlobsBox(2);
x2 = x1 + thisBlobsBox(3);
y2 = y1 + thisBlobsBox(4);

%fprintf(1,'#%d %.1f %.1f %.1f %.1f\n', k, x1, x2, y1, y2);
x = [x1 x2 x2 x1 x1];
y = [y1 y1 y2 y2 y1];
%subplot(3,4,2);
plot(x, y, 'LineWidth', 2);

%neck region removed
h=(y2-y1)-.25*(y2-y1);
y3=[x1 y1 x2-x1 h];
i2=imcrop(imgh,y3);
figure,imshow(i2);

%lips template
y4=[x1 y1+(2*h/3) x2-x1 h/3];
i3=imcrop(imgh,y4);
figure,imshow(i3);

s = fspecial('sobel');
figure,imshow(imfilter(i3, s));

%eyes template
y5=[x1 y1+(h/6) x2-x1 (h/3)+10];
i4=imcrop(imgh,y5);
figure,imshow(i4);

figure,imshow(imfilter(i4, s));
%%
end
meanImage = mean(img,2);        
img = (img - meanImage*ones(1,numTestImage))';
Projected_Test = img*C;

%% ########################################################################


%% ################# Calculation of Distance from Neutral ##################
meanNutral = mean(S(NeutralImages,EigenRange)',2);
for Dat2Project = 1:numTestImage
    TestImage = Projected_Test(Dat2Project,:);
    % Picking the image #Dat2Project
 
    Eucl_Dist(Dat2Project) = sqrt((TestImage'-meanNutral)'*(TestImage' ...
        -meanNutral));
        % Here, the distance between the expression under test and
        % the mean neutral expressions is being calculated
end
%Eucl_Dist = Eucl_Dist/max(Eucl_Dist);
%% ########################################################################

%% ################# Calculation of other Distances #######################
Other_Dist = zeros(numTestImage,numImage);
for Dat2Project = 1:numTestImage
    TestImage = Projected_Test(Dat2Project,:);
    % Picking the image #Dat2Project
    for i = 1:numImage
        Other_Dist(Dat2Project,i) = sqrt((TestImage'-S(i,EigenRange)')' ...
            *(TestImage'-S(i,EigenRange)'));
    end
end
[Min_Dist,Min_Dist_pos] = min(Other_Dist,[],2);
%% ########################################################################


%% ########################## Display Result ##############################
fid = fopen('Results.txt','w');
fprintf(fid,'//Test Image,Distance From Neutral, Expression,Best Match\r\n');

for i = 2:numTestImage
    b = find(TestImages{i,1}=='\');
    Test_Image = TestImages{i,1}(b(end)+1:end);
    Dist_frm_Neutral = Eucl_Dist(i);
    Best_Match = cell2mat(imageLabel{1,1}(Min_Dist_pos(i)));
    Expr = cell2mat(imageLabel{1,2}(Min_Dist_pos(i)));
    if(strcmp(Expr,'disgust'))
        setappdata(0,'music',Expr);
        %[y,Fs] = wavread('11.wav');
        %player = audioplayer(y, Fs);
        %play(player);
         
    end
    if(strcmp(Expr,'happy'))
        setappdata(0,'music',Expr);
        %[y,Fs] = wavread('11.wav');
        %player = audioplayer(y, Fs);
        %play(player);
         
    end
    if(strcmp(Expr,'anger'))
        setappdata(0,'music',Expr);
        %[y,Fs] = wavread('11.wav');
        %player = audioplayer(y, Fs);
        %play(player);
    end
    if(strcmp(Expr,'sad'))
        setappdata(0,'music',Expr);
        %[y,Fs] = wavread('11.wav');
        %player = audioplayer(y, Fs);
        %play(player);
    end
    act_ive_gui_new();
    fprintf(fid,'%s,%0.0f,%s,%s\r\n',Test_Image,Dist_frm_Neutral,Expr,Best_Match);
    
end
fclose(fid);
%% ########################################################################
isSucceed = 1;
disp('Done')
disp('Output File = .\Results.txt');
winopen('Results.txt');
Willexit = input('Press Enter to Quit ...','s');
end