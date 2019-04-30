#include "try.h"
#include "preemptiveSLIC.h"
#include <iostream>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;

int main()
{
	//----------------------------------序列图像读取---------------------------------

	int imageAmount = 0;									//图像序列数
	vector<Mat> srcImage;									//存储原始图像
	char* Path = "D:/Visual Studio/dataset/unpacked//bottle/bottle (";
	char* Postfix = ").jpg";

	imageAmount = readImageSequence(Path, Postfix, srcImage);

	cout << "读取图像done" << endl;

	vector<Mat> srcGrayImage;                                                //存储灰度图像

	for (int i = 0; i < imageAmount; i++)
	{
		Mat temp;
		cvtColor(srcImage[i], temp, COLOR_BGR2GRAY);
		srcGrayImage.push_back(temp.clone());
	}

	//----------------------------------Shi-Tomasi角点检测---------------------------------

	vector<Point2f> firstPoint;		                                  //存储检测到的角点

	cornerDetection(srcGrayImage[0], firstPoint);

	cout << "检测角点done" << endl;

	//---------------------LK金字塔光流生成点轨迹矩阵---------------------												                             


	Mat W;				//轨迹矩阵
	int range = 5;		//对前range帧追踪光流
	vector<vector<Point2f> > rawPoint(range, vector<Point2f>(0));

	trajectoryGeneration(srcGrayImage, firstPoint, range, rawPoint, W);

	cout << "点轨迹生成done" << endl;

	//重新存储实际检测到的角点
	vector<vector<Point2f> > Point(range, vector<Point2f>(0));
	storePoint(W, range, Point);

	cout << "存储角点done" << endl;

	//----------------------------------SSC聚类---------------------------------

	//正则化参数
	double Lambda1;
	Lambda1 = computeLambda(W);

	cout << "正则化参数done" << endl;

	//仿射空间系数矩阵
	Mat C2;
	C2 = admmLasso(W, Lambda1);

	cout << "仿射空间系数矩阵done" << endl;

	//thrC
	Mat cP;
	cP = thrC(C2);

	cout << "thrC done" << endl;

	//BuildAdjacency
	Mat CKSym;
	CKSym = buildAdjacency(cP);
	cout << "CKSym done" << endl;

	//SpectralClustering
	Mat Label1;
	int n = 2;		//聚类数
	Label1 = spectralClustering(CKSym,n);

	cout << "点轨迹聚类done" << endl;

	//确定前背景标记
	uchar foreG = 0;
	uchar backG = 0;

	label(Label1, foreG, backG);

	cout << "确定前背景标记done" << endl;
	
	//----------------------------------ORB追踪---------------------------------

	vector<vector<Point2f> > featurePoint(imageAmount, vector<Point2f>(0));          //存储角点特征点
	vector<vector<Point2f> > orbPoint(imageAmount, vector<Point2f>(0));			     //存储ORB特征点
	vector<vector<float> > featureCor(imageAmount, vector<float>(0));				 //存储特征点坐标极值
	vector<vector<float> > orbCor(imageAmount, vector<float>(0));					 //存储ORB特征点极值

	 //计算图像序列特征点和区域
	for (size_t i = 0; i < Point[0].size(); i++)
	{
		if (Label1.at<int>(i, 0) == foreG)
			featurePoint[0].push_back(Point[0][i]);
	}

	track(srcGrayImage, featurePoint, orbPoint, featureCor, orbCor);

	cout << "追踪done" << endl;


	//----------------------------------超像素生成---------------------------------

	int Review = 2;										    //显示图像帧
	float maxFX, maxFY, minFX, minFY;                       //区域边界
	maxFX = featureCor[Review][0] + 20;
	minFX = featureCor[Review][1] - 20;
	maxFY = featureCor[Review][2] + 20;
	minFY = featureCor[Review][3] - 20;


	Rect rec(minFX, minFY, maxFX - minFX, maxFY - minFY);
	Mat Roi(srcImage[Review], rec);
	//Mat Roi = srcImage[Review](Range(minFY, maxFY + 1), Range(minFX, maxFX + 1));   //超像素区域

	// parameters
	Mat tt = imread("D:/disk.jpg");
	int  Segment = 50;                                    //希望的超像素数
	double Compactness = 50;
	int superPixelNum = 0;								    //超像素数目

	Mat seeds;
	int *labels_preemptiveSLIC;
	PreemptiveSLIC preemptiveSLIC;
	//preemptiveSLIC.preemptiveSLIC(Roi, Segment, Compactness*2.5, labels_preemptiveSLIC, seeds);
	preemptiveSLIC.preemptiveSLIC(tt, Segment, Compactness*2.5, labels_preemptiveSLIC, seeds);

	// label image L
	//int dimx = Roi.cols;
	//int dimy = Roi.rows;
	int dimx = tt.cols;
	int dimy = tt.rows;

	Mat L;
	fillMat(L, labels_preemptiveSLIC, dimx, dimy);

	int maxLab = 0;				//最大标记值
	for (int i = 0; i < dimy; i++)
		for (int j = 0; j < dimx; j++)
		{
			if (L.at<int>(i, j) > maxLab)
				maxLab = L.at<int>(i, j);
		}
	superPixelNum = maxLab + 1;

	// boundary image B
	Mat B;
	getBoundaryImage(B, L, dimx, dimy);

	// overlay image          
	Mat R;
	//getOverlayedImage(R, B, Roi);
	getOverlayedImage(R, B, tt);

	//计算超像素坐标
	vector<vector<Point2i>> Coordinate(superPixelNum, vector<Point2i>(0));
	computeCoordinate(L, superPixelNum,  dimx,  dimy, Coordinate);

	/*//验证超像素坐标存储正确性
	Mat check(dimy, dimx, CV_8UC3,Scalar(255,255,255));
	for (int k = 0; k < superPixelNum; k++)
	{
		for (size_t i = 0; i < Coordinate[k].size(); i++)
		{
			int row = Coordinate[k][i].x;
			int col = Coordinate[k][i].y;
			check.at<Vec3b>(row, col) = (0, 0, 10 * k);
		}
	}
	namedWindow("hh", WINDOW_NORMAL);
	namedWindow("hhh", WINDOW_NORMAL);
	imshow("hh", check);
	imshow("hhh", R);
	waitKey(0);
	*/

	//计算每一个超像素包含的像素数
	vector<int> numOfEachSuperpixel;
	for (int i = 0; i < superPixelNum; i++)
	{
		int temp = (int)Coordinate[i].size();
		numOfEachSuperpixel.push_back(temp);
	}
	

	cout << "超像素生成done" << endl;

	//----------------------------------提取颜色特征---------------------------------

	//RGB转HSV
	Mat hsvImage;			//存储HSV图像
	Mat gray;				//存储灰度图像
	//cvtColor(Roi, hsvImage, COLOR_BGR2HSV);
	//cvtColor(Roi, gray, COLOR_BGR2GRAY);
	cvtColor(tt, hsvImage, COLOR_BGR2HSV);
	cvtColor(tt, gray, COLOR_BGR2GRAY);
	vector<Mat> hsvSuperPixelVector;		   //存储hsv图像超像素

	 //计算HSV超像素容器
	buildHsvSuperPiexlVector(Coordinate, numOfEachSuperpixel, hsvImage, superPixelNum, hsvSuperPixelVector);

	//参数准备
	int hueBinNum = 9;
	int saturationBinNum = 8;
	int valueBinNum = 6;
	vector<Mat> colorFeature;

	//计算颜色特征向量
	ColorFeature(hsvSuperPixelVector, hueBinNum, saturationBinNum,  valueBinNum,  superPixelNum, colorFeature);

	cout << "颜色特征提取done" << endl;

	//----------------------------------提取纹理特征---------------------------------

	//计算Iwld值
	Mat IwldValue = Mat::zeros(dimy, dimx, CV_64F);

	computeIwld(gray, dimx, dimy, IwldValue);

	//标准化Iwld到[0，255]
	Mat normIwld = Mat::zeros(dimy, dimx, CV_64F);
	Mat NormIwld = Mat::zeros(dimy, dimx, CV_8U);
	normalize(IwldValue, normIwld, 0, 255, NORM_MINMAX);

	for (int i = 0; i < dimy; i++)
	{
		for (int j = 0; j < dimx; j++)
		{
			NormIwld.at<uchar>(i, j) = cvRound(normIwld.at<double>(i, j));
		}
	}

	//计算标准化Iwld超像素容器

	vector<Mat> textureFeatureVector;
	buildIwldVector( Coordinate,  numOfEachSuperpixel, NormIwld, superPixelNum, textureFeatureVector);

	//计算纹理特征向量

	vector<Mat> textureFeature;
	TextureFeature(textureFeatureVector, superPixelNum, textureFeature);

	cout << "提取纹理特征done" << endl;

	//----------------------------------超像素聚类---------------------------------

	//计算颜色空间距离
	Mat colorDistance = Mat::zeros(superPixelNum, superPixelNum, CV_64F);
	double sumOfColDis = 0;

	computeColorDis(colorFeature, superPixelNum, colorDistance, sumOfColDis);
	

	//计算超纹理空间距离
	Mat textureDistance = Mat::zeros(superPixelNum, superPixelNum, CV_64F);
	double sumOfTexDis = 0;

	computeTexDis(textureFeature, superPixelNum, textureDistance, sumOfTexDis);

	
	//计算曼哈顿距离 

	Mat  manDistance = Mat::zeros(superPixelNum, superPixelNum, CV_64F);
	double sumOfManDis = 0;

	computeManDis(Coordinate, numOfEachSuperpixel, superPixelNum, manDistance);

	//归一化曼哈顿距离
	double maxManDis = 0;                           //最大曼哈顿距离
	for (int i = 0; i < superPixelNum; i++)
		for (int j = i; j < superPixelNum; j++)
		{
			if (manDistance.at<double>(i, j) > maxManDis)
				maxManDis = manDistance.at<double>(i, j);
		}

	manDistance = manDistance / maxManDis;

	for (int i = 0; i < superPixelNum; i++)
		for (int j = i; j < superPixelNum; j++)
		{
			sumOfManDis += manDistance.at<double>(i, j);
		}

	//各距离权重
	double wclolor = sumOfColDis / (sumOfColDis + sumOfTexDis + sumOfManDis);
	double wtexture = sumOfTexDis / (sumOfColDis + sumOfTexDis + sumOfManDis);
	double wman = sumOfManDis / (sumOfColDis + sumOfTexDis + sumOfManDis);

	cout << "wclolor = " << wclolor << endl;
	cout << "wtexture = " << wtexture << endl;
	cout << "wman = " << wman << endl;
	
	//计算超像素距离

	Mat superPixelDis;
	superPixelDis = wclolor*colorDistance + wtexture*textureDistance + wman*manDistance;

	//计算西格玛
	double Xi = 0;
	for (int i = 0; i < superPixelNum; i++)
	{
		double temp = 1;
		for (int j = 0; j < superPixelNum; j++)
		{
			if (j == i)
				continue;
			else if (superPixelDis.at<double>(i, j) < temp)
				temp = superPixelDis.at<double>(i, j);
		}
		if (i == 0)
			Xi = temp;
		else if (temp > Xi)
			Xi = temp;
	}
	cout << "Xi = " << Xi << endl;

	//计算权重矩阵
	Mat Weight = Mat::zeros(superPixelNum, superPixelNum, CV_64F);

	computeWeight(superPixelDis, superPixelNum, Xi, Weight);

	//计算度矩阵D
	Mat DN(superPixelNum, superPixelNum, CV_64F, Scalar(0));
	computeD(Weight, superPixelNum, DN);

	//计算拉普拉斯矩阵
	Mat LapN;
	Mat eye = Mat::eye(Size(superPixelNum, superPixelNum), CV_64F);
	LapN = eye - DN * Weight * DN;

	//计算特征值、特征向量
	Mat eValuesMat;
	Mat teVectorsMat, eVectorsMat;

	eigen(LapN, eValuesMat, teVectorsMat);
	eVectorsMat = teVectorsMat.t();

	int numOfCluster = 20;

	//计算Y矩阵
	Rect region(0, 0, numOfCluster, superPixelNum);
	Mat V(eVectorsMat, region);

	Mat vtY = Mat::zeros(superPixelNum, numOfCluster, CV_32F);
	computeV(V, numOfCluster, superPixelNum, vtY);

	Mat Label2;
	TermCriteria Termcriteria1(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	double compactness = kmeans(vtY, numOfCluster, Label2, Termcriteria1, 3, KMEANS_PP_CENTERS);
	
	cout << "超像素聚类done" << endl;

	//超像素填充
	Mat seg(dimy, dimx, CV_8UC3, Scalar::all(255));

	pixelCluster(Coordinate, numOfEachSuperpixel, Label2, superPixelNum, seg);

	cout << "超像素填充done" << endl;

	Mat h;
	//namedWindow("jj", WINDOW_NORMAL);
	//namedWindow("hh", WINDOW_NORMAL);
	imshow("hh", R);
	getOverlayedImage(h, B, seg);
	imshow("jj", h);
	imshow("hhh", hsvImage);
	imshow("hhhhh", NormIwld);
	waitKey(0);

	return 0;
}

	//图像前背景标记

	//Mat Extract(srcImage[Review].rows, srcImage[Review].cols, CV_8UC3, Scalar(0,255,255));
	//Mat Extract(dimy,dimx ,CV_8UC3, Scalar(0, 255, 255));


	/*
	Mat countP(superPixelNum, 1, CV_8U, Scalar::all(0));			//每一个超像素中有的特征点数
    Mat Label3(superPixelNum, 1, CV_8U, Scalar::all(0));		    //每一类超像素的标记。0：前景，1：不确定
	Mat eachCluster(numOfCluster, 2, CV_8U, Scalar::all(0));		//每一类超像素分别标记为前背景的数量


	for (int j = 0; j < featurePoint[Review].size(); j++)
	{
		int row = (int)featurePoint[Review][j].y;
		int col = (int)featurePoint[Review][j].x;

		if (row >= 0)
		{
			int pixel = B.at<int>(row, col);
			int pixelR = B.at<int>(row, col + 1);
			int pixelD = B.at<int>(row + 1, col);

			if (pixel == 0)
			{
				countP.at<uchar>(pixel, 0) += 1;
			}
			else if (pixel != pixelR)
			{
				countP.at<uchar>(pixelR, 0) += 1;
				countP.at<uchar>(pixel, 0) += 1;
			}

			else
			{
				countP.at<uchar>(pixelD, 0) += 1;
				countP.at<uchar>(pixel, 0) += 1;
			}
		}

	}
	*/
	/*
	for (int i = 0; i < superPixelNum; i++)
	{
		if (countP.at<uchar>(i, 0) > 0)
		{
			for (int j = 0; j < numOfEachSuperpixel[i]; j++)
			{
				int x = Coordinate[i][j].x;
				int y = Coordinate[i][j].y;
				Extract.at<Vec3b>(x, y)[0] = 0;
				Extract.at<Vec3b>(x, y)[1] = 0;
				Extract.at<Vec3b>(x, y)[2] = 255;
			}
			int temp = Label2.at<int>(i, 0);
			eachCluster.at<uchar>(temp, 0) += 1;
			Label3.at<uchar>(i, 0) = 0;
		}
		else
			/*
			for (int j = 0; j < numOfEachSuperpixel[i]; j++)
			{
				int x = Coordinate[i][j].x;
				int y = Coordinate[i][j].y;
				Extract.at<Vec3b>(x, y)[0] = 0;
				Extract.at<Vec3b>(x, y)[1] = 255;
				Extract.at<Vec3b>(x, y)[2] = 255;
			}
			*/
	/*
		{
			int temp = Label2.at<int>(i, 0);
			eachCluster.at<uchar>(temp, 1) += 1;
			Label3.at<uchar>(i, 0) = 1;
		}
	}
	*/
	/*
	for (int i = 0; i < superPixelNum; i++)
	{
		if (Label3.at<uchar>(i, 0) == 1)
		{
			int temp = Label2.at<int>(i, 0);
			if (eachCluster.at<uchar>(temp, 0) > eachCluster.at<uchar>(temp, 1))
				for (int j = 0; j < numOfEachSuperpixel[i]; j++)
				{
					int x = Coordinate[i][j].x ;
					int y = Coordinate[i][j].y ;
					Extract.at<Vec3b>(x, y)[0] = 0;
					Extract.at<Vec3b>(x, y)[1] = 0;
					Extract.at<Vec3b>(x, y)[2] = 255;
				}
		}
	}

	*/

	/*
	int r = 1;
	for (size_t i = 0; i < featurePoint[Review].size(); i++)
	{
		circle(Extract, featurePoint[Review][i], r, Scalar(255, 0, 0), -1, 8, 0);
		//circle(srcImage[k], Point[k][i], r, Scalar(0, 0, 255), -1, 8, 0);
		//Rect rec0(featureCor[Review][1], featureCor[k][3], featureCor[k][0] - featureCor[k][1], featureCor[k][2] - featureCor[k][3]);
		//Rect rec1(orbCor[k][1], orbCor[k][3], orbCor[k][0] - orbCor[k][1], orbCor[k][2] - orbCor[k][3]);
		//rectangle(srcImage[k], rec0, Scalar(0, 255, 255));
		//rectangle(srcImage[k], rec1, Scalar(255, 255, 0));
	}

	for (size_t i = 0; i < orbPoint[Review].size(); i++)
	{
		circle(Extract, orbPoint[Review][i], r, Scalar(255, 255,0), -1, 8, 0);
	}

	
	
	imshow("hhh", R);
	waitKey(0);
	
	*/
//	imshow("hh", Extract);
	//waitKey(0);
	//return 0;






	/*
	//int r = 1;
	for (int k = 0; k <imageAmount; k++)
	{
		for (size_t i = 0; i < featurePoint[k].size(); i++)
		{
			circle(srcImage[k], featurePoint[k][i], r, Scalar(0, 255, 0), -1, 8, 0);
			//circle(srcImage[k], Point[k][i], r, Scalar(0, 0, 255), -1, 8, 0);
			Rect rec0(featureCor[k][1], featureCor[k][3], featureCor[k][0] - featureCor[k][1], featureCor[k][2] - featureCor[k][3]);
			Rect rec1(orbCor[k][1], orbCor[k][3], orbCor[k][0] - orbCor[k][1], orbCor[k][2] - orbCor[k][3]);
			rectangle(srcImage[k], rec0, Scalar(0, 255, 255));
			rectangle(srcImage[k], rec1, Scalar(255, 255, 0));
		}

		for (size_t i = 0; i < orbPoint[k].size(); i++)
		{
			circle(srcImage[k], orbPoint[k][i], r, Scalar(0, 0, 255), -1, 8, 0);
		}
	}

	for (int i = 0; i < imageAmount; i++)
	{
		stringstream filenameS;
		filenameS << "featurePoint" << i << ".png";
		//imshow(filenameS.str(), srcImage[i]);
	}

	*/