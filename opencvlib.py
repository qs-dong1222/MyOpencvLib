### -*- coding: utf-8 -*-

import numpy as np;
import cv2;
import time;
import random;
import math;
import operator

COLOR_BLUE = (255,0,0);
COLOR_GREEN = (0,255,0);
COLOR_RED = (0,0,255);
COLOR_WHITE = (255,255,255);
COLOR_BLACK = (0,0,0);


# ---------------------------------  Decorators ---------------------------------------#
# count execution time
def TimeCountWrapper(funcName):
	def RealWrapper(*args):
		e1 = cv2.getTickCount();
		ret = funcName(*args);
		e2 = cv2.getTickCount();
		print ("executime time of function", \
			   funcName.__name__, " = ", \
			   (e2 - e1) / cv2.getTickFrequency());
		if(ret.all() != None): # ret is an array. has to use all()
			return ret;
	return RealWrapper;
# ---------------------------------  Decorators ---------------------------------------#






# ---------------------------------  control ---------------------------------------#
def WaitEscToExit():
	while True:
		if(cv2.waitKey(1)&0xFF == 27):
			print("quit");
			break;
	cv2.destroyAllWindows();


def IfEscKeyDown():
    if (cv2.waitKey(1) & 0xFF == 27):
        return True;
    else:
        return False;


def IfKeyDown(keyChar):
    if (cv2.waitKey(1) & 0xFF == ord(keyChar)):
        return True;
    else:
        return False;




# Get variable name
def GetVarName(a):
    import inspect, re;
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bGetVarName\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line);
    if m:
        return m.group(1);
# ---------------------------------  control ---------------------------------------#



# See if image is colorful
def IsImageColorful(img):
    return (True if(len(img.shape)>2) else False);





@TimeCountWrapper
def GetImgMask(imgPath, filterThreshold = 220):
    Img = cv2.imread(imgPath);
    imgGray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY);
    unuse, mask = cv2.threshold(imgGray,filterThreshold,255,cv2.THRESH_BINARY_INV);
    mask_inv = cv2.bitwise_not(mask);
    return mask_inv;


@TimeCountWrapper
def TransparentPaint(img_bg, img_fg, x, y, transparentExtent=220):
    rows,cols,channels = img_fg.shape; # get fg pic demension
    roi = img_bg[x:x+rows, y:y+cols]; # get region from bg according to fg pic demension
    imgGray = cv2.cvtColor(img_fg, cv2.COLOR_BGR2GRAY); # get gray fg pic 
    unuse, mask_inv = cv2.threshold(imgGray,transparentExtent,255,cv2.THRESH_BINARY_INV); # get mask_inv(white content black bkgnd)
    mask = cv2.bitwise_not(mask_inv); # get mask(black content white bkgnd)
    bg = cv2.bitwise_and(roi,roi,mask=mask); # black out content area in roi
    fg = cv2.bitwise_and(img_fg,img_fg,mask=mask_inv); # black out bkgnd in original pic
    dst = cv2.add(bg,fg); # combine the 2 above, get content with bkgnd
    img_bg[x:x+rows, y:y+cols] = dst;
    return img_bg;
	

# 定义平移Shift函数
@TimeCountWrapper
def Shift(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # 返回转换后的图像
    return shifted
	
	
	
# 定义旋转rotate函数, angle>0 顺时针旋转, angle<0 逆时针旋转
@TimeCountWrapper
def Rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[0:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, -angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated
	
	

	
# 创建插值方法数组
cv2InterpolMethods = {
    "cv2.INTER_NEAREST": cv2.INTER_NEAREST,
    "cv2.INTER_LINEAR": cv2.INTER_LINEAR,
    "cv2.INTER_AREA": cv2.INTER_AREA,
    "cv2.INTER_CUBIC": cv2.INTER_CUBIC,
    "cv2.INTER_LANCZOS4": cv2.INTER_LANCZOS4
}
# 1.在一般情况下， cv2.INTER_NEAREST速度最快，但质量不高。如果资源非常有限，可以考虑使用。否则不选这个，尤其在上采样（增加）图像的大小时。
# 2.当增加（上采样）图像的大小时，可以考虑使用 cv2.INTER_LINEAR 和 cv2.INTER_CUBIC两种插值方法。 cv2.INTER_LINEAR 方法比 cv2.INTER_CUBIC
# 	方法稍快，但无论哪个效果都不错。
# 3.当减小（下采样）的图像的大小，OpenCV的文档建议使用
#   cv2.INTER_AREA。PS. 感觉其实下采样各种方法都差不多，取速度最快的（cv2.INTER_NEAREST）就好。

# 定义缩放resize函数
#@TimeCountWrapper
def Resize(image, width=None, height=None, cv2InterpolMethod=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
    dim = None;
    (h, w) = image.shape[:2];
    # 如果宽度和高度均为0，则返回原图
    if(width is None and height is None):
        return image;
    elif height>h or width>w:
        cv2InterpolMethod = cv2.INTER_CUBIC;
    dim = (width,height);
    resized = cv2.resize(image, dim, interpolation=cv2InterpolMethod);
    # 返回缩放后的图像
    return resized;


""" 
def Resize(image, width=None, height=None, cv2InterpolMethod=cv2.INTER_AREA):
    # 初始化缩放比例，并获取图像尺寸
	dim = None
	(h, w) = image.shape[:2]
    # 如果宽度和高度均为0，则返回原图
	if width is None and height is None:
		return image
    # 宽度是0
	if width is None:
        # 则根据高度计算缩放比例
		r = height / float(h);
		if(r > 1):
			cv2InterpolMethod = cv2.INTER_CUBIC;
		dim = (int(w * r), height)
    # 如果高度为0
	else:
        # 根据宽度计算缩放比例
		r = width / float(w);
		if(r > 1):
			cv2InterpolMethod = cv2.INTER_CUBIC;
		dim = (width, int(h * r))
    # 缩放图像
	resized = cv2.resize(image, dim, interpolation=cv2InterpolMethod)
    # 返回缩放后的图像
	return resized
"""












	
	
	

MirrorDirections = {
	"HORIZONTAL": 1,
	"VERTICAL": 0
}
# 定义镜像翻折Mirror函数
@TimeCountWrapper
def Mirror(image, mirrorDirection = MirrorDirections["HORIZONTAL"]):
	flipped = cv2.flip(image, mirrorDirection);
	return flipped;
	
	

# 加边框函数	
def MakeBorder(image, borderWidth=10, borderColor=(255,255,255)):
    imgExtend = cv2.copyMakeBorder(image, borderWidth,\
                                   borderWidth, borderWidth, \
                                   borderWidth, cv2.BORDER_CONSTANT, \
                                   borderColor);
    return imgExtend;
	




# affine transformation
# pts[0] --------- pts[1]
#        |      /
#        |     /
#        |    /
#        |   /
#        |  /
#        | /
#        |/
#       pts[2]
#
# retval - affine Img
def AffineTransform(image,ori3pts,target3pts):
	h, w = image.shape[0:2];
	oriPts = np.float32([ ori3pts[0],ori3pts[1],ori3pts[2] ]);
	tarPts = np.float32([ target3pts[0],target3pts[1],target3pts[2] ]);
	M = cv2.getAffineTransform(oriPts, tarPts);
    # 3rd param - output image size
	affineImg = cv2.warpAffine(image, M, (w, h));
	return affineImg;
	
	





# Perspective Transform
# pts[0]/left top ------------- pts[1]/right top
#                 |           |
#                 |           |
# pts[3]/left bot ------------- pts[2]/right bot
# ori4pts[] - pts[0]/left top, pts[1]/right top,
#             pts[2]/right bot, pts[3]/left bot
# target4pts[] - pts[0]/left top, pts[1]/right top,
#                pts[2]/right bot, pts[3]/left bot
# retval - Perspective Img
def PerspectiveTransform(image,ori4pts,target4pts):
    h,w = image.shape[0:2];
    pts1 = np.float32([ ori4pts[0],ori4pts[1],ori4pts[2],ori4pts[3] ]);
    pts2 = np.float32([ target4pts[0],target4pts[1],target4pts[2],target4pts[3] ]);
    M = cv2.getPerspectiveTransform(pts1,pts2);
    PerspectiveImg = cv2.warpPerspective(image,M,tuple([w,h]));
    return PerspectiveImg;






	
# param - mophShape: cv2.MORPH_CROSS
#                    cv2.MORPH_ELLIPSE
#                    cv2.MORPH_RECT
# retval - kernel indicated
# get a 0,1 matrix based on shape
def GetAreaKernel(mophShape, row=3,col=3):
    shape = (col,row);
    if(mophShape == cv2.MORPH_CROSS):
        return cv2.getStructuringElement(cv2.MORPH_CROSS, shape);
    elif(mophShape == cv2.MORPH_ELLIPSE):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, shape);
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, shape);






# get outline by cv2.canny()
# param - image: input image
#       - outlineDensity: higher is density, more is the outline edge
# retval - outline image
def GetCannyOutlineImage(image, outlineDensity):
    return cv2.Canny(image, 0, outlineDensity);








# get otus single-channel gray image
# param - a general image, does not matter it is colorful or not
# retval - bestThreshold: best value of threshold
#        - otusized image
def GetOtusImage(img):
    if(IsImageColorful(img)): # colorful image
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    bestThreshold, otusImg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU);
    return (bestThreshold,otusImg);



# pixel value greater than threshold will convert to white
# retval - retImg: processed gary image
def GrayGTthresh2White(imgGray, threshold=100):
    ret, imgThresh = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY);
    return imgThresh;



# pixel value greater than threshold will convert to black
# retval - retImg: processed gary image
def GrayGTthresh2Black(imgGray, threshold=100):
    ret, imgThresh = cv2.threshold(imgGray, threshold, 255, cv2.THRESH_BINARY_INV);
    return imgThresh;




# color image convert to gray image
# retval - gray image
def GetGrayImage(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);





# find pixels from a gray image
# example - FindCoordsInImg(imgGray, imgGray>threshold)
def FindCoordsInGrayImg(imgGray,condition):
    return np.transpose(np.where(condition)[::-1]);






# get contours from a image
# param - imgOriginal: original image
#       - threshold: 0~255, threshold gray depth used to seperate/distinguish contour
#       - isImgWhiteBkgnd: if the image's background color is white
# retval - contourList: contour list
def GetContours(img, threshold, isImgWhiteBkgnd=False):
    if(IsImageColorful(img)):
        img = GetGrayImage(img);
    if(isImgWhiteBkgnd):
        imgThreshed = GrayGTthresh2Black(img, threshold);
    else:
        imgThreshed = GrayGTthresh2White(img, threshold);
    unusedImg, contourList, hierarchy = cv2.findContours(imgThreshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);
    return contourList;






# get only those outer contours from a image, ignoring the enclosed inner parts
# param - imgOriginal: original image
#       - threshold: 0~255, threshold gray depth used to seperate/distinguish contour
#       - isImgWhiteBkgnd: if the image's background color is white
# retval - contourList: contour list
def GetOuterContours(img, threshold,isImgWhiteBkgnd=False):
    if (IsImageColorful(img)):
        img = GetGrayImage(img);
    if (isImgWhiteBkgnd):
        imgThreshed = GrayGTthresh2Black(img, threshold);
    else:
        imgThreshed = GrayGTthresh2White(img, threshold)
    unusedImg, contourList, hierarchy = cv2.findContours(imgThreshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    return contourList;






# draw all contours on image
# retval - originalImg: copy of source image. This function will modify the source image
#        - contoursImg: image with contours
def DrawAllContoursOnImage(RGBimgToBeDrawn,contourList,contourColor=COLOR_GREEN,drawPointSize=2):
    originalImg = RGBimgToBeDrawn.copy();
    contoursImg = cv2.drawContours(RGBimgToBeDrawn, contourList, -1, contourColor, drawPointSize);
    return originalImg,contoursImg;




# draw a single specific contour on image
# param - contourList: list of contour found by GetContours()
#       - contourIndex: the indicated index in contour list to be drawn
# retval - originalImg: copy of source image. This function will modify the source image
#        - contoursImg: image with contours
def DrawContourOnImage(RGBimgToBeDrawn,contourList,contourIndex,contourColor=COLOR_GREEN,drawPointSize=2):
    originalImg = RGBimgToBeDrawn.copy();
    contoursImg = cv2.drawContours(RGBimgToBeDrawn, contourList, contourIndex, contourColor, drawPointSize);
    return originalImg,contoursImg;






# get barycenter inside a contour
def GetContourBarycore(contour):
    M = cv2.moments(contour);
    cx = int(M['m10'] / M['m00']);
    cy = int(M['m01'] / M['m00']);
    return (cx,cy);

# get area of a contour
def GetContourArea(contour):
    return cv2.contourArea(contour);

# get Perimeter of a contour
def GetContourPerimeter(contour,isContourClosed=True):
    return cv2.arcLength(contour,isContourClosed);




# Get a restored-convexed-out contour
# param - contour: a contour to be convexd out
#       - isContourClosed: is this contour to be closed out
# retval - repairContour: a list of contour points that can be drawn already
def GetRepairContour(contour, polygonNbr, isContourClosed=True):
    found = False;
    for epsilonCoeff in np.arange(0, 1, 0.01):
        epsilon = epsilonCoeff * GetContourPerimeter(contour);
        repairContour = cv2.approxPolyDP(contour, epsilon, isContourClosed);
        # 是否为凸
        if (cv2.isContourConvex(repairContour)):
            # 看是几边形
            if(len(repairContour)==polygonNbr):
                found = True;
                break;
    if(found):
        return [repairContour];
    else:
        return None;




# find Convex points in contour List
# retval - ConvexList: list of Convexes for each single contour in contour List
def FindConvexInContours(contourList):
    ConvexList = [];
    for eachContour in contourList:
        ConvexList.append( cv2.convexHull(eachContour, returnPoints=True) );
    return ConvexList;




""" 
##########################################
# FUNCTION: STILL FIND CONVEX BUT CLOW
##########################################
# find Concave points in contour List
# retval - ConcaveList: list of Concaves for each single contour in contour List
def FindConcaveInContours(contourList):
    concaveIndexGroup = [];
    concaveList = [];
    for eachContour in contourList:
        concaveIndexGroup.append(cv2.convexHull(eachContour, returnPoints=False));
        tempPointIndexList = concaveIndexGroup[-1].flatten();
        tempPointList = list(map(lambda eachIndex:eachContour[eachIndex],tempPointIndexList));
        tempConcaves = tempPointList[0];
        for eachPoint in tempPointList:
            tempConcaves = np.vstack( (tempConcaves,eachPoint) );
        concaveList.append(tempConcaves);
    return concaveList;
"""



# find a non-optmized rect contour based on a input contour points list
# param - contour: a single contour in contour list
# retval - rectContour: list of points of rect contour
def FindRectContour(contour):
    x, y, w, h = cv2.boundingRect(contour);
    topleft = np.array([ [x,y] ]);
    topright = np.array([ [x+w,y] ]);
    bottomright = np.array([ [x+w,y+h] ]);
    bottomleft = np.array([ [x,y+h] ]);
    rectContour = np.concatenate( (topleft,topright,bottomright,bottomleft),axis=0 );
    rectContour = rectContour[np.newaxis, :]; # increase a demension to 3 demensions
    return rectContour;




# find a optmized rect contour points based on a input contour points list
# using cv2.minAreaRect(contour), it returns a Box2D structure which contains
# rect center(x，y), rect width and height, rotating angle
# param - contour: a single contour in contour list
# retval - rectContourPoints: list of points of rect contour
def FindOptRectContour(contour):
    rectContourPoints = [];
    rect = cv2.minAreaRect(contour);
    rectContourPoints.append( np.int0(cv2.boxPoints(rect)) );
    return rectContourPoints;







# get a mask(white foreground, black background) based on contourList in a image
# param - oriImg: the image where the contour is found
#         contourList: a contour
# retval - contourMask: the mask image
def GetContourRegionMask(oriImg, contourList):
    h,w = oriImg.shape[0:2];
    contourMask = GetBlankImg(w,h);
    cv2.drawContours(contourMask, contourList, -1, 255, -1);
    return contourMask;






# find enclosing circle to a single contour
# param - contour: a single contour in contour list
# retval - center: center of circle
#        - radius: radius of circle
def FindEnclosingCircle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour);
    center = (int(x), int(y));
    radius = int(radius);
    return center,radius;






# find the contour with max area from given contour list
# param - contourList: contour list found by GetContours()
# retval - maxContourAreaIndex: concerning index of max area contour in contour list
#        - maxContourArea: max area of the found contour
def GetMaxAreaContourFromContourList(contourList):
    areaList = list(map(lambda eachConour:GetContourArea(eachConour), contourList));
    sortedConDict = sorted(dict([(index, eachArea) for index, eachArea in zip(range(0, len(areaList)), areaList)]).items(),
           key=lambda eachItem: eachItem[1],reverse=True);
    maxContourAreaIndex = sortedConDict[0][0];
    maxContourArea = sortedConDict[0][1];
    return maxContourAreaIndex,maxContourArea;




# Get a blank(black) gray-scale image
# sizeTuple = (height, width)
def GetBlankImg( w,h ):
    sizeTuple = (h,w);
    return np.zeros(sizeTuple, np.uint8);



# find the coordinates where the non-black pixels locate
# retval - list of coordinates
def FindNonBlackInGrayImg(imgGray):
    pixelCoords = cv2.findNonZero(imgGray);
    return pixelCoords;
"""
np.where() or np.nonzero() also can be used after transpose

>>> x = np.eye(3)
>>> x
array([ [ 1., 0., 0.],
        [ 0., 1., 0.],
        [ 0., 0., 1.] ])
>>> np.nonzero(x)
( array([0, 1, 2]), 
  array([0, 1, 2]) )
>>> x[np.nonzero(x)]
array([ 1., 1., 1.])
>>> np.transpose(np.nonzero(x))
array([ [0, 0],
        [1, 1],
        [2, 2] ])
"""



# find most top, most bottom, most left, most right, 4 coords in a contour
# param - contour: a single contour
# retval - list of [mostTopCoord,mostBottomCoord,mostLeftCoord,mostRightCoord]
def FindClimaxCoordInContour(contour):
    allXcoords = contour[:, :, 0];
    allYcoords = contour[:, :, 1];
    leftXindex = allXcoords.argmin();
    rightXindex = allXcoords.argmax();
    topYindex = allYcoords.argmin();
    bottomYindex = allYcoords.argmax();
    mostLeftCoord = contour[leftXindex][0];
    mostRightCoord = contour[rightXindex][0];
    mostTopCoord = contour[topYindex][0];
    mostBottomCoord = contour[bottomYindex][0];
    return [mostTopCoord,mostBottomCoord,mostLeftCoord,mostRightCoord];






# Find defects among all the convex region in contour
# param - contour: a single contour
# retval - defectList: list of defect info, each info contains (convex start pt coord),
#                                                              (convex end pt coord),
#                                                              (concave pt coord),
#                                                              (distance to concave)
def FindConvexDefects(contour):
    hull = cv2.convexHull(contour, returnPoints=False);
    defects = cv2.convexityDefects(contour, hull);
    defectList = [];
    for i in range(defects.shape[0]):
        s, e, f, distance = defects[i, 0];
        tempList = [];
        start = tuple(contour[s][0]);
        tempList.append(start);
        end = tuple(contour[e][0]);
        tempList.append(end);
        far = tuple(contour[f][0]);
        tempList.append(far);
        tempList.append(distance);
        defectList.append(tempList);
    return defectList;




# see if a point is outside a contour
def IfPixelOutOfContour(contour, pixelCoord):
    ret = cv2.pointPolygonTest(contour, pixelCoord, False);
    if(ret == -1):
        return True;
    else:
        return False;

# see if a point is on the edge of a contour
def IfPixelOnContour(contour, pixelCoord):
    if(cv2.pointPolygonTest(contour, pixelCoord, False) == 0 ):
        return True;
    else:
        return False;


# distance from a point to the edge of a contour
def DistancePixelToContour(contour, pixelCoord):
    return cv2.pointPolygonTest(contour, pixelCoord, True);






# find 4 climaxes coordinates from a TEXT image
# param - binary_img: binary image with white foreground
# retval - list of 4 climax coordinates
def FindRotatedImgClimaxes(binary_img):
    topleft = TopLeftCornerScan(binary_img);
    topright = TopRightCornerScan(binary_img);
    botleft = BotLeftCornerScan(binary_img);
    botright = BotRightCornerScan(binary_img);
    return [topleft,topright,botright,botleft]






# distance between tow pixels
def Distance(pixel_1,pixel_2):
    return math.sqrt( math.pow(pixel_1[0]-pixel_2[0],2) + math.pow(pixel_1[1]-pixel_2[1],2) );





# calibrate the rotated image
# param - binary_img: binary image with white foreground
# retval - caliAngle: angle should be calibrated
#        - topleft: corner coordinate before calibration
#        - topright: corner coordinate before calibration
#        - botright: corner coordinate before calibration
#        - botleft: corner coordinate before calibration
def RotatedImgCalibrate(binary_img):
    climaxes = FindRotatedImgClimaxes(binary_img);
    # 让程序自主进行四个角的标号
    cont = np.array(list(map(lambda x:list(x), climaxes)));
    rectContour = FindOptRectContour(cont);
    rectMask = GetContourRegionMask(binary_img,rectContour);
    contour = GetOuterContours(rectMask, 100, False);
    rect = cv2.minAreaRect(contour[0]);
    center, size, angle = rect;
    rect_w = size[0];
    rect_h = size[1];
    if(-2<=angle<=0 or -88<=angle<=-90):
        print("no rotation at all");
        caliAngle = 0
    else:
        vertices = np.int0(cv2.boxPoints(rect));
        dist_0_1 = Distance(vertices[0],vertices[1]);
        # distance between no.0 and no.1 is width
        if(rect_w-2 <dist_0_1 < rect_w+0):
            if(angle>=-45):
                topleft = rectContour[0][1];
                topright = rectContour[0][2];
                botright = rectContour[0][3];
                botleft = rectContour[0][0];
                caliAngle = -angle
            else:
                topleft = rectContour[0][2];
                topright = rectContour[0][3];
                botright = rectContour[0][0];
                botleft = rectContour[0][1];
                caliAngle = -(90+angle)
        # distance between no.0 and no.1 is height
        else:
            if (angle >= -45):
                topleft = rectContour[0][1];
                topright = rectContour[0][2];
                botright = rectContour[0][3];
                botleft = rectContour[0][0];
                caliAngle = -angle
            else:
                topleft = rectContour[0][2];
                topright = rectContour[0][3];
                botright = rectContour[0][0];
                botleft = rectContour[0][1];
                caliAngle = -(90 + angle)

    return caliAngle  #,topleft,topright,botright,botleft





    # top = climaxes[0];
    # bottom = climaxes[1];
    # left = climaxes[2];
    # right = climaxes[3];
    # width = Distance(bottom, right);
    # height = Distance(bottom, left);
    # if(IsImgWitheBkgnd):
    #     imgNot = cv2.bitwise_not(img);
    #     bestThreshold, imgOtus = GetOtusImage(imgNot);
    # else:
    #     bestThreshold, imgOtus = GetOtusImage(img);
    # pxList = FindCoordsInGrayImg(imgOtus, imgOtus > 0);
    # center, size, angle = cv2.minAreaRect(pxList);
    # if (width > height):
    #     # clockwise rotation
    #     return -angle;
    # else:
    #     # counter clockwise rotation
    #     return -90-angle;















# find stright lines in image by Probability Hough Transformation
# param - image: original image
#       - targetLineNum: how many lines you want to extract
#       - minLineLength: min length(uint in pixel) to be considered as a line in image
#       - maxLineGap: how far the gap is, for which 2 discontinuous lines in same direction
#                     can be considered as a continuoues line
# retval - lineX1Y1X2Y2group: group of head and end points for each line that has
#                             been detected
#        - lineImg: image that contains only those line found
def GetHoughLines(image, targetLineNum, minLineLength, maxLineGap):
    if(IsImageColorful(image)):
        image = GetGrayImage(image);
    i = 0;
    while(True):
        # By given targetLineNum, dynamically adjust min&max threshold
        # in Canny edge detection and Hough Transformation until the
        # given line number matched
        otusEdgesImg = cv2.Canny(image, i, i+2, apertureSize=3, L2gradient=True);
        lineX1Y1X2Y2group = cv2.HoughLinesP(otusEdgesImg, 1, np.pi / 180,
                                threshold=100,
                                minLineLength=minLineLength,
                                maxLineGap=maxLineGap);
        lineNum = len(lineX1Y1X2Y2group);

        # found those lines under given targetLineNum
        if(lineNum<=targetLineNum):
            lineImg = GetBlankImg(image.shape);
            for eachPtsGroup in lineX1Y1X2Y2group:
                cv2.line(lineImg, (eachPtsGroup[0][0], eachPtsGroup[0][1]),
                                   (eachPtsGroup[0][2], eachPtsGroup[0][3]),
                                   (255, 255, 255), 1);
            return lineX1Y1X2Y2group, lineImg;

        # speed up the filtering process
        elif(lineNum>=targetLineNum*5):
            i += 30;
        elif(targetLineNum*2<=lineNum<targetLineNum*5):
            i += 10;
        else:
            i += 1;








# find intersections among given lines
# param - img: original image
#       - lineX1Y1X2Y2group: lineX1Y1X2Y2group: group of head and end points for each line that has
#                            been given
# retval - interPts: intersection group that has been found
def FindLineIntersections(img,lineX1Y1X2Y2group):
    lineGroup = [];
    blankImg = GetBlankImg(img.shape[0:2]);
    # Given couple of lines head&end, we expand them to let them intersect with each other
    for eachX1Y1X2Y2 in lineX1Y1X2Y2group:
        # build up full lines from lines head&end using slope and intercept
        x1, y1, x2, y2 = eachX1Y1X2Y2[0];
        # print("x1, y1, x2, y2",x1, y1, x2, y2);

        if(x1==x2 and y1!=y2):
            # vertical line
            k = float(math.tan(3.1415/2));
            b = y1 - k*x1;
        elif(y1==y2 and x1!=x2):
            # horizontal line
            k = 0;
            b = y1;
        else:
            coeff = np.array([[x1, 1], [x2, 1]]);
            res = np.array([y1, y2]);
            # print("coeff", coeff);
            # print("res", res);
            k, b = np.linalg.solve(coeff, res);
        lineGroup.append(tuple([k, b]));

        # draw lines to reveal intersections on a blank image
        xPts = [0, img.shape[1]];
        yPts = list(map(lambda x: int(k * x + b), xPts));
        cv2.line(blankImg, (xPts[0], yPts[0]),
                 (xPts[1], yPts[1]),
                 COLOR_WHITE, 1);

    lineNum = len(lineGroup);
    interPts = [];
    # calculate intersections among all lines from stright line expression
    # delete those incorrect intersections by verifying the pixel value exsitence
    # take 2 lines to calculate below every time
    for index, line1_KBval in enumerate(lineGroup[0:lineNum - 1]):
        k1 = line1_KBval[0];
        b1 = line1_KBval[1];
        # print("out", k1, b1);
        for line2_KBval in lineGroup[index + 1:]:
            k2 = line2_KBval[0];
            b2 = line2_KBval[1];
            # print("in", k2, b2);
            if(k1==0 and k2!=0):
                interY = int(b1);
                interX = int(round((b1-b2)/k2));
            elif(k2==0 and k1!=0):
                interY = int(b2);
                interX = int(round((b2-b1)/k1));
            elif(k2==0 and k1==0):
                # parallel lines
                pass;
            elif(k1 == k2):
                # parallel lines
                pass;
            else:
                coeff = np.array([[k1, -1], [k2, -1]]);
                b_res = np.array([-b1, -b2]);
                interX, interY = (np.linalg.solve(coeff, b_res));
                interX = int(interX);
                interY = int(interY)
            # check if these intersections are inside the image
            if (0 <= interX < img.shape[1] and 0 <= interY < img.shape[0]):
                if (blankImg[interY, interX] != 0):
                    interPts.append(tuple([interX, interY]));

    # erase same intersections
    interPts = list(set(interPts));
    return interPts;







# get a Histogram
# param - img: the image where histogram extract
#       - mask: mask shade to mask out the region unconcerned
#       - ifHoldResult: if store the Histogram result to next Histogram
# retval - hist: Histogram
def GetHistogram(img, isImgColorful=False, colorToSatistic='r',
                 mask=None, ifHoldResult=False):
    if(not isImgColorful):
        hist = cv2.calcHist([img], [0], mask, [256], [0, 256]);
    elif(colorToSatistic == 'r'):
        print("red")
        hist = cv2.calcHist([img], [2], mask, [256], [0, 256]);
    elif(colorToSatistic == 'g'):
        print("green")
        hist = cv2.calcHist([img], [1], mask, [256], [0, 256])
    else:
        print("blue")
        hist = cv2.calcHist([img], [0], mask, [256], [0, 256]);
    return hist;





# global histogram equalization to enhance image contrast
def HistGlobalEqualize(img):
    if (IsImageColorful(img)):
        b, g, r = cv2.split(img);
        b_equ = cv2.equalizeHist(b);
        g_equ = cv2.equalizeHist(g);
        r_equ = cv2.equalizeHist(r);
        img_equ = cv2.merge([b_equ, g_equ, r_equ]);
    else:
        img_equ = cv2.equalizeHist(img);
    return img_equ;







# optimal(Regional) histogram equalization to enhance image contrast
def HistOptimalEqualize(img, kernelSize=8):
    if (IsImageColorful(img)):
        b, g, r = cv2.split(img);
        b_clahe_equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernelSize,kernelSize)).apply(b);
        g_clahe_equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernelSize,kernelSize)).apply(g);
        r_clahe_equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernelSize,kernelSize)).apply(r);
        img_equ = cv2.merge([b_clahe_equ, g_clahe_equ, r_clahe_equ]);
    else:
        img_equ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernelSize,kernelSize)).apply(img);
    return img_equ;









# fill up a region with new color at a coordinate soecified
def FillUpConnectRegion(img, coordInRegion, color):
    oriImg = img.copy();
    h, w = img.shape[0:2];
    retval, filledImg, mask, rect = cv2.floodFill(img,mask=None,seedPoint=coordInRegion,newVal=color);
    return oriImg,filledImg;




# count how many pixel appears on a image column
def VerticalPixelCount(img,columnIndex):
    if(IsImageColorful(img)):
        img = GetGrayImage(img);
    colList = np.where(img > 0)[::-1][0].tolist();
    return colList.count(columnIndex);


# count how many pixel appears on a image row
def HorizontalPixelCount(img,rowIndex):
    if (IsImageColorful(img)):
        img = GetGrayImage(img);
    rowList = np.where(img > 0)[::-1][1].tolist();
    return rowList.count(rowIndex);







#test if a coordinate is in image
def IfCoordOutOfImage(x,y, img):
    h, w = img.shape[0:2];
    if( x<0 or
        x>=w or
        y<0 or
        y>=h):
        return True;
    else:
        return False;


# search for top left existed  pixel
def TopLeftCornerScan(img):
    h, w = img.shape[0:2];
    for iter_w in range(w):
        for x,y in zip(range(iter_w, -1, -1), range(h)):
            if(not IfCoordOutOfImage(x,y,img)):
                if(img[y,x]!=0):
                    return tuple([x,y]);
    for iter_h in range(h):
        for x,y in zip(range(w-1,-1,-1), range(iter_h,h)):
            if(not IfCoordOutOfImage(x, y, img)):
                if (img[y, x] != 0):
                    return tuple([x,y]);

# search for bottom left existed  pixel
def BotLeftCornerScan(img):
    h, w = img.shape[0:2];
    for iter_w in range(w):
        for x,y in zip(range(iter_w, -1, -1), range(h,-1,-1)):
            if(not IfCoordOutOfImage(x,y,img)):
                if(img[y,x]!=0):
                    return tuple([x,y]);
    for iter_h in range(h,-1,-1):
        for x,y in zip(range(w-1,-1,-1), range(iter_h,-1,-1)):
            if(not IfCoordOutOfImage(x, y, img)):
                if (img[y, x] != 0):
                    return tuple([x,y]);

# search for bottom right existed  pixel
def BotRightCornerScan(img):
    h, w = img.shape[0:2];
    for iter_w in range(w,-1,-1):
        for x,y in zip(range(iter_w, w), range(h,-1,-1)):
            if(not IfCoordOutOfImage(x,y,img)):
                if(img[y,x]!=0):
                    return tuple([x,y]);
    for iter_h in range(h,-1,-1):
        for x,y in zip(range(0,w), range(iter_h,-1,-1)):
            if(not IfCoordOutOfImage(x, y, img)):
                if (img[y, x] != 0):
                    return tuple([x,y]);

# search for top right existed  pixel
def TopRightCornerScan(img):
    h, w = img.shape[0:2];
    for iter_w in range(w,-1,-1):
        for x,y in zip(range(iter_w, w), range(h)):
            if(not IfCoordOutOfImage(x,y,img)):
                if(img[y,x]!=0):
                    return tuple([x,y]);
    for iter_h in range(h):
        for x,y in zip(range(0,w), range(iter_h,h)):
            if(not IfCoordOutOfImage(x, y, img)):
                if (img[y, x] != 0):
                    return tuple([x,y]);





# clean those contours from contour list on area raito with respect to image area
# param - img: original image
#       - contourList: contour list
#       - minAreaRatio: min area ratio to delete contour
#       - maxAreaRatio: max area ratio to delete contour
# retval - cleaned contour list
def CleanContoursByArea(contourList,minAreaInPixel,maxAreaInPixel):
    # h, w = img.shape[0:2]
    # imgArea = h * w;
    idxes = [];
    for idx, c in enumerate(contourList):
        area = GetContourArea(c);
        if (area <= minAreaInPixel
            or area >= maxAreaInPixel):
            pass
        else:
            idxes.append(idx);
    newContourList = [];
    for idx in idxes:
        newContourList.append(contourList[idx]);
    contourList = newContourList;
    return contourList;






# clean those contours from contour list based on ratio between
# long axis and short axis of fit ellipse
# param - contourList: contour list
#       - min_long_short_ratio: min a/b ratio to delete contour
#       - max_long_short_ratio: max a/b ratio to delete contour
# retval - cleaned contour list
def CleanContoursByEllipseAxisRatio(contourList,min_long_short_ratio,max_long_short_ratio):
    idxes = [];
    for idx, c in enumerate(contourList):
        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        if(MA<ma):
            MA,ma = ma,MA
        if(ma==0):
            ma=0.1
        long_short_ratio = MA/ma
        if (long_short_ratio <= min_long_short_ratio
            or long_short_ratio >= max_long_short_ratio):
            pass
        else:
            idxes.append(idx);
    newContourList = [];
    for idx in idxes:
        newContourList.append(contourList[idx]);
    contourList = newContourList;
    return contourList;













def MSER(img,
         delta=5,
         minArea2del=60,
         maxArea2del=14400,
         maxVariation=0.25,
         minDiversity=0.2,
         MaxEvolution=200,
         AreaThreshold=1.01,
         MinMargin=0.003,
         EdgeBlurSize=5,
         Mask=None):
# param - delta
# indicates through how many
# different gray levels does a region need
# to be stable to be considered maximally stable.
# For a larger delta, you will get less regions.

# param - minArea, maxArea, unit is in pixel
# If a region is maximally stable,
# it can still be rejected if it has less than minArea pixels or more than maxArea pixels.

# param - maxVariation
# it is like a tolerance about the changed area of water surface when water level goes up
# if this tolerance is high, means bigger change on water surface will be accepted, namely more regions will be saved.
# if a region is maximally stable, it can still be rejected if the the regions
# variation is bigger than maxVariation. That is, even if the region is "relatively"
# stable (more stable than the neigbouring regions), it may not be "absolutely" stable enough.
# For smaller maxVariation, you will get less regions


# param - minDiversity
# This parameter exists to prune regions that are too similar (e.g. differ for only a few pixels).
# For a region Region_T1 that is maximally stable with threshold T1(gray/water level), find a region
# Region_T2 which is the "parent maximally stable region" with bigger threshold T2(gray/water level).
# That means, T2 > T1, Region_T2 is a maximally stable region and there is no maximally stable region between
# T2 > Tx > T1. Now, compare how much bigger the parent region(Region_T2) is:
#
# diversity = (size(Region_T2) - size(Region_T1)) / size(Region_T1), i.e. Diff(AreaT2,AreaT1)/AreaT1
#
# If this diversity is smaller than maxDiversity, remove the region CC_T1(p). For larger diversity,
# you will get less regions.

# other params:
# MaxEvolution - for color image, the evolution steps. default 200.
# AreaThreshold - the area threshold to cause re-initialize. default 1.01.
# MinMargin - ignore too small margin. default 0.003.
# EdgeBlurSize - the aperture size for edge blur. default 5.
# Mask Optional - input mask that marks the regions where we should detect features
    mser = cv2.MSER_create(_delta=delta,
                           _min_area=minArea2del,
                           _max_area=maxArea2del,
                           _max_variation=maxVariation,
                           _min_diversity=minDiversity,
                           _max_evolution=MaxEvolution,
                           _area_threshold=AreaThreshold,
                           _min_margin=MinMargin,
                           _edge_blur_size=EdgeBlurSize);
    if(Mask is not None):
        img = cv2.bitwise_and(img,img,mask=Mask);
    regions, _ = mser.detectRegions(img);
    del mser;
    return regions;












# test if a coordinate is at boundary of an image
def IfCoordAtBoundary(img,coord):
    x, y = coord;
    left = [x - 1,y];
    lettop = [x - 1,y - 1];
    right = [x + 1,y];
    righttop = [x + 1,y - 1];
    top = [x,y - 1];
    leftbot = [x - 1,y + 1];
    bot = [x,y + 1];
    rightbot = [x + 1,y + 1];
    eightNeiborPx = [right, righttop, top, lettop, left, leftbot, bot, rightbot];
    for coord in eightNeiborPx:
        if(IfCoordOutOfImage(coord[0],coord[1],img)):
            return True
    return False

# get a list of coordinates in side a circle
def GetCoordsInCircle(img, centroid, radius):
    h, w = img.shape[0:2]
    cir_mask = GetBlankImg(w, h)
    cv2.circle(cir_mask, centroid, radius, COLOR_WHITE, -1)
    cv2.imshow("cir_mask",cir_mask)
    coordx, coordy = np.where(cir_mask != 0)[::-1]
    coords = [[x, y] for x, y in zip(coordx, coordy)]
    return coords;

# get a list of coordinates inside a rect
def GetCoordsInRect(img, centroid, rect_w, rect_h):
    h, w = img.shape[0:2]
    centroid_x = centroid[0];
    centroid_y = centroid[1];
    rect_mask = GetBlankImg(w, h)
    cv2.rectangle(rect_mask, (int(centroid_x-rect_w/2),int(centroid_y-rect_h/2)), (int(centroid_x+rect_w/2),int(centroid_y+rect_h/2)), COLOR_WHITE, -1);
    cv2.imshow("rect_mask",rect_mask)
    coordx, coordy = np.where(rect_mask != 0)[::-1]
    coords = [[x, y] for x, y in zip(coordx, coordy)]
    return coords;

# calculate the gradient(mag,theta) of a coord according to its neighborhood
def CalcNeighborGradientAtCoord(img,coord):
    x,y = coord;
    left = float(img[y,x-1]);
    right = float(img[y, x + 1]);
    top = float(img[y - 1, x]);
    bot = float(img[y + 1, x]);

    Gx = (left-right)/2.0
    Gy = (bot-top)/2.0

    if(Gx==0):
        # theta < 0, top is higher than bot
        # theta > 0, top is lower than bot
        # theta = 0, top = bot
        tmp_theta = (math.atan(Gy) * 180) / math.pi;
        if(Gy>0):
            theta = tmp_theta;
        else:
            theta = tmp_theta + 360
    elif(Gy==0):
        # theta < 0, right is higher than left
        # theta > 0, left is lower than right
        # theta = 0, left = right
        tmp_theta = (math.atan(1/Gx) * 180) / math.pi;
        if(Gx>0):
            theta = tmp_theta;
        else:
            theta = tmp_theta + 180;
    else:
        tmp_theta = (math.atan(Gy/Gx) * 180) / math.pi;
        if(Gx>0 and Gy>0):
            theta = tmp_theta;
        elif(Gx<0 and Gy>0):
            theta = tmp_theta + 180;
        elif(Gx>0 and Gy<0):
            theta = tmp_theta +360;
        else:
            theta = tmp_theta + 180
    # gradient magnitude
    Gmag = math.sqrt(math.pow(Gx,2) + math.pow(Gy,2))
    return Gmag,theta

# get a main gradient direction in a circle region
# param - thetaResolution: the resolution you want to subdevide angle 0 ~ 360
def GetCirAreaMainGradient(img, centroid, radius, thetaResolution=36):
    # get coords in circle
    coords = GetCoordsInCircle(img, centroid, radius);
    # calculate highest neighborhood and lowest neighborhood
    # for each coord in 8 neighborhoods, then group them
    Gmags = []
    thetas = []
    for centroid in coords:
        if(IfCoordAtBoundary(img,centroid)):
            pass
        else:
            Gmag, theta = CalcNeighborGradientAtCoord(img, centroid)
            Gmags.append(Gmag);
            thetas.append(theta);
    if (np.any(Gmags) == 0):
        main_Gtheta = 0.0;
        main_Gmag = 0.0;
        return main_Gtheta, main_Gmag

    # partition angle from 0 to 360 to subgroups based on thetaResolution
    # e.g. thetaResolution=30, it would be [0,30,60,...,330,360]
    # then sum all magnitude within each angle range and obtain dominant gradient theta
    step = thetaResolution;
    upperBound = 360;
    lowerBound = 0;
    theta_range = np.linspace(lowerBound,upperBound,int(360/step)+1)
    theta_mag_dict = {};
    for bound_a in theta_range[:-1]:
        bound_b = bound_a + step;
        # print(bound_a,bound_b)###############
        mag_sum_in_bounds = 0;
        for idx,theta in enumerate(thetas):
            if(bound_a <= theta <bound_b):
               mag_sum_in_bounds += Gmags[idx];
        theta_mag_dict[str(bound_a)] = mag_sum_in_bounds;
    theta_mag_dict = sorted(theta_mag_dict.items(), key=operator.itemgetter(1), reverse=True);
    # print(theta_mag_dict)##########
    main_Gtheta = float(theta_mag_dict[0][0]);
    main_Gmag = float(theta_mag_dict[0][1]);
    return main_Gtheta,main_Gmag

# get a main gradient direction in a circle region
# param - thetaResolution: the resolution you want to subdevide angle 0 ~ 360
def GetRectAreaMainGradient(img, centroid, rect_w, rect_h, thetaResolution=36):
    # get coords in rect
    coords = GetCoordsInRect(img, centroid, rect_w, rect_h)
    # calculate highest neighborhood and lowest neighborhood
    # for each coord in 8 neighborhoods, then group them
    Gmags = []
    thetas = []
    for centroid in coords:
        if(IfCoordAtBoundary(img,centroid)):
            pass
        else:
            Gmag, theta = CalcNeighborGradientAtCoord(img, centroid)
            Gmags.append(Gmag);
            thetas.append(theta);
    if(np.any(Gmags)==0):
        main_Gtheta = 0.0;
        main_Gmag = 0.0;
        return main_Gtheta, main_Gmag

    # partition angle from 0 to 360 to subgroups based on thetaResolution
    # e.g. thetaResolution=30, it would be [0,30,60,...,330,360]
    # then sum all magnitude within each angle range and obtain dominant gradient theta
    step = thetaResolution;
    upperBound = 360;
    lowerBound = 0;
    theta_range = np.linspace(lowerBound,upperBound,int(360/step)+1)
    theta_mag_dict = {};
    for bound_a in theta_range[:-1]:
        bound_b = bound_a + step;
        # print(bound_a,bound_b)###############
        mag_sum_in_bounds = 0;
        for idx,theta in enumerate(thetas):
            if(bound_a <= theta <bound_b):
               mag_sum_in_bounds += Gmags[idx];
        theta_mag_dict[str(bound_a)] = mag_sum_in_bounds;
    theta_mag_dict = sorted(theta_mag_dict.items(), key=operator.itemgetter(1), reverse=True);
    # print(theta_mag_dict)##############
    main_Gtheta = float(theta_mag_dict[0][0]);
    main_Gmag = float(theta_mag_dict[0][1]);
    return main_Gtheta, main_Gmag
