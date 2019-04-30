#include "try.h"

using namespace std;
using namespace cv;


//---------------------求正则化参数--------------------


double computeLambda(const Mat & W)
{
	Mat T, tW;
	int N = W.cols;
	double lambda1 = 0;
	tW = W.t();
	T = tW * W;

	for (int i = 0; i < N; i++)
	{
		T.at<double>(i, i) = 0;
	}

	T = abs(T);

	for (int j = 0; j < N; j++)
	{
		double temp = 0;
		for (int i = 0; i < N; i++)
		{
			if (T.at<double>(i, j) > temp)
				temp = T.at<double>(i, j);
		}
		if (j == 0)
			lambda1 = temp;
		else if (temp < lambda1)
			lambda1 = temp;
	}
	return lambda1;
}

//---------------------仿射空间系数矩阵--------------------
Mat admmLasso(const Mat& W, const double Lambda1)

{
	double Alpha = 800;
	double Thr = 2e-4;
	int maxIter = 200;
	double err1, err2;
	int N = W.cols;

	//为ADMM设置惩罚参数

	double mu1 = Alpha / Lambda1;
	double mu2 = Alpha;

	//初始化
	Mat A, iA, eye, Rone, Cone, one, C1, C2,lambda2, lambda3, Z, tW;
	eye = Mat::eye(Size(N, N), CV_64F);
	one = Mat::ones(Size(N, N), CV_64F);
	Cone = Mat::ones(Size(1, N), CV_64F);		//列向量
	Rone = Mat::ones(Size(N, 1), CV_64F);       //行向量
	tW = W.t();
	iA = mu1*(tW*W) + mu2*eye + mu2*one;
	A = iA.inv();
	C1 = Mat::zeros(Size(N, N), CV_64F);
	lambda2 = Mat::zeros(Size(N, N), CV_64F);
	lambda3 = Mat::zeros(Size(N, 1), CV_64F);
	int it = 0;
	err1 = 10 * Thr;
	err2 = 10 * Thr;

	//ADMM迭代

	while ((err1 > Thr || err2 > Thr) && it < maxIter)
	{
		//更新Z

		Z = A * (mu1*(tW*W) + mu2*(C1 - lambda2 / mu2) + mu2*Cone*(Rone - lambda3 / mu2));
		for (int i = 0; i < N; i++)
		{
			Z.at<double>(i, i) = 0;
		}

		//更新C

		Mat temp1, temp2;
		temp1 = Z + lambda2 / mu2;
		temp1 = abs(temp1);
		temp1 = temp1 - 1 / mu2 * one;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (temp1.at<double>(i, j) < 0)
					temp1.at<double>(i, j) = 0;
			}
		}
		temp2 = Z + lambda2 / mu2;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (temp2.at<double>(i, j) < 0)
					temp2.at<double>(i, j) = -1;
				else if (temp2.at<double>(i, j) > 0)
					temp2.at<double>(i, j) = 1;
				else
					temp2.at<double>(i, j) = 0;
			}
		}
		C2 = temp1.mul(temp2);

		for (int i = 0; i < N; i++)
		{
			C2.at<double>(i, i) = 0;
		}

		//更新拉格朗日算子

		lambda2 = lambda2 + mu2 * (Z - C2);
		lambda3 = lambda3 + mu2 * (Rone * Z - Rone);

		//计算误差

		Mat temp3;
		temp3 = Z - C2;
		temp3 = abs(temp3);

		for (int j = 0; j < N; j++)
		{
			double k = 0;
			for (int i = 0; i < N; i++)
			{
				if (temp3.at<double>(i, j) > k)
					k = temp3.at<double>(i, j);
			}
			if (j == 0)
				err1 = k;
			else if (k > err1)
				err1 = k;
		}

		Mat temp4;
		temp4 = Rone * Z - Rone;
		temp4 = abs(temp4);
		double k = 0;
		for (int j = 0; j < N; j++)
		{
			if (temp4.at<double>(0, j) > k)
				k = temp4.at<double>(0, j);
		}
		err2 = k;

		//赋值

		C1 = C2;
		it++;
	}

	return C2;
}

//---------------------thrC--------------------
Mat thrC(const Mat & C2)
{
	double rho = 0.7;
	int N = C2.cols;
	Mat C, absC, Ind, S;
	Mat cP = Mat::zeros(Size(N, N), CV_64F);
	C = C2.clone();
	absC = abs(C);
	cv::sort(absC, S, SORT_EVERY_COLUMN + SORT_DESCENDING);
	sortIdx(absC, Ind, SORT_EVERY_COLUMN + SORT_DESCENDING);

	for (int j = 0; j < N; j++)
	{
		double cL1 = 0;
		for (int i = 0; i < N; i++)
		{
			cL1 += S.at<double>(i, j);
		}
		bool stop = false;
		double cSum = 0;
		int t = 0;
		while (!stop)
		{
			cSum += S.at<double>(t, j);
			t++;
			if (cSum >= rho * cL1)
			{
				stop = true;
				for (int k = 0; k < t; k++)
				{
					int temp = Ind.at<int>(k, j);
					cP.at<double>(temp, j) = C.at<double>(temp, j);
				}
			}
		}
	}
	return cP;
}

//---------------------buildAdjacency--------------------
Mat buildAdjacency(const Mat& cP)
{
	int N = cP.cols;
	double eps = 2.2204e-16;
	Mat CAbs, Srt, tCAbs, cksym, Ind,CKSym;
	CAbs = abs(cP);
	cv::sort(CAbs, Srt, SORT_EVERY_COLUMN + SORT_DESCENDING);
	sortIdx(CAbs, Ind, SORT_EVERY_COLUMN + SORT_DESCENDING);

	for (int j = 0; j < N; j++)
	{
		int temp0 = Ind.at<int>(0, j);
		double temp = CAbs.at<double>(temp0, j) + eps;
		CAbs.col(j) = CAbs.col(j) / temp;
	}

	tCAbs = CAbs.t();
	CKSym = CAbs + tCAbs;

	return CKSym;
}

//SpectralClustering
Mat spectralClustering(const Mat &CKSym,const int n)
{
	int N = CKSym.cols;
	double eps = 2.2204e-16;
	Mat eye = Mat::eye(Size(N, N), CV_64F);
	Mat DN(N, N, CV_64F, Scalar(0));
	Mat rSum(1, N, CV_64F, Scalar(0));
	reduce(CKSym, rSum, 0, REDUCE_SUM);
	TermCriteria Termcit(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
	rSum += eps;

	for (int j = 0; j < N; j++)
	{
		double temp = sqrt(rSum.at<double>(0, j));
		DN.at<double>(j, j) = 1 / temp;
	}

	Mat LapN, uN, sN, tvN, vN;
	LapN = eye - DN * CKSym * DN;
	SVD::compute(LapN, sN, uN, tvN);
	vN = tvN.t();
	int rnum = vN.rows;
	Rect rec(N - n, 0, n, rnum);
	Mat kerN(vN, rec);
	Mat kerNS(kerN.size(), CV_32F);
	for (int i = 0; i < N; i++)
	{
		double temp = norm(kerN.row(i), NORM_L2) + eps;
		for (int j = 0; j < n; j++)
		{
			kerNS.at<float>(i, j) = (float)(kerN.at<double>(i, j) / temp);
		}
	}

	Mat label;
	vector<Point2f> centers;
	double compactness = kmeans(kerNS, n, label, Termcit, 3, KMEANS_PP_CENTERS, centers);

	return label;
}

//确定前背景标记
void label(const Mat &Label, uchar &foreG, uchar &backG)
{
	int sl1 = 0;
	int el1 = 0;
	int sl2 = 0;
	int el2 = 0;

	for (int i = 0; i < Label.rows; i++)
	{
		if (Label.at<int>(i, 0) == Label.at<int>(0, 0))
			el1 = i;
		else if (sl2 == el2)
			sl2 = i;
		else
			el2 = i;
	}
	if ((el1 - sl1) > (el2 - sl2))
	{
		backG = Label.at<int>(sl1, 0);
		foreG = Label.at<int>(sl2, 0);
	}
	else
	{
		backG = Label.at<int>(sl2, 0);
		foreG = Label.at<int>(sl1, 0);
	}
}
