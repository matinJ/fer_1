#include "Fer.h"
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <io.h>
#include <iostream>

using namespace std;

#define DATA_PATH "..\\fer_easy_data"

double Fer::run(double c)
{
	//训练
	SVMParams params;
	params.svm_type = SVM::C_SVC;
	params.C = c;
	params.kernel_type = SVM::LINEAR;
	params.term_crit = TermCriteria(CV_TERMCRIT_EPS, 0, 5e-3);

	SVM svm;
	svm.train(_TrainFeature, _TrainLabels, Mat(), Mat(), params);

	//测试

	int train_WA = 0, test_WA = 0;
	for (int i = 0; i < _TrainFeature.rows; i++)
	{
		Mat sample = _TrainFeature.row(i);
		int response = static_cast<int>(svm.predict(sample));
		int label = static_cast<int>(_TrainLabels.at<float>(i, 0));
		if (response == label)
			train_WA++;
		else
			test_WA++;
	}

	FILE* fp = fopen("..//fer.txt", "a+");
	fprintf(fp, "================================\n");
	fprintf(fp, "fer train rate: %.2lf\n", (double)train_WA / (train_WA + test_WA));
	fprintf(fp, "svm support vector count: %d\n", svm.get_support_vector_count());

	const float* pram0 = svm.get_support_vector(0);

	int WA = 0, AC = 0;
	for (int i = 0; i < _TestFeature.rows; i++)
	{
		Mat sample = _TestFeature.row(i);
		int response = static_cast<int>(svm.predict(sample));
		int label = static_cast<int>(_TestLabels.at<float>(i, 0));
		if (response == label)
			AC++;
		else
			WA++;
	}

	fprintf(fp, "fer test rate: %.2lf with c : %.9lf\n\n", (double)AC / (AC + WA), c);

	fclose(fp);

	return (double)AC / (AC + WA);
}

void Fer::ExtrateImgFeature(Mat &img, Mat& featureMat, Size dsize,
	Size winSize, Size blockSize,
	Size blockStride, Size cellSize,
	int nbins, int derivAperture, double winSigma,
	int histogramNormType, double L2HysThreshold,
	bool gammaCorrection, int nlevels)
{
	assert(img.cols == 32 && img.rows == 32);

	// HOGDescriptor hog(winSize, blockSize,
	//	blockStride, cellSize, nbins, derivAperture,
	//	winSigma, histogramNormType, L2HysThreshold,
	//	gammaCorrection, nlevels);
	//std::vector<float> descriptors;
	//hog.compute(img, descriptors);

	//if (featureMat.rows == 0)
	//	featureMat =  Mat(0, descriptors.size(), CV_32FC1);

	// Mat feature =  Mat(descriptors).t();

	Mat re_mat = img.reshape(0, 1);
	Mat fl_mat = Mat(1, re_mat.cols, CV_32FC1);
	for (int i = 0; i < re_mat.cols; i++)
		fl_mat.at<float>(0, i) = re_mat.at<uchar>(0, i);

	if (featureMat.rows == 0)
		featureMat = Mat(0, re_mat.cols, CV_32FC1);

	featureMat.push_back(fl_mat);

}

/*
std::string relPath2AbsPath(std::string path)
{
	char moduleFileName[_MAX_PATH];
	GetModuleFileName(NULL, moduleFileName, _MAX_PATH);
	String str(moduleFileName);
	int pos = str.ReverseFind('\\');
	str = str.Left(pos + 1);
	return std::string(StringA(str).GetBuffer()) + path;
}
*/
void Fer::HandleFeatureAndLabels(int is_pca)
{
	//初始化
	int train_sample_count[7] = { 600, 250, 250, 450, 200, 250, 250 };
	//2600*2304(48*48)

	_TrainLabels = Mat(0, 1, CV_32FC1);
	_TestLabels = Mat(0, 1, CV_32FC1);

	//从文件夹中读图片，提取特征，放到内存中
	int trian_count = 0;
	for (int ClassNum = 0; ClassNum < 7; ClassNum++)
	{
		char ClassStr = ClassNum + '0';
		std::string data_path0(DATA_PATH);
		data_path0 = data_path0 + ClassStr;
		string data_path = data_path0 + "\\*";

		_finddata_t FileInfo;
		long Handle = _findfirst(data_path.c_str(), &FileInfo);
		if (Handle == -1L)
		{
			cerr << "can't match the path" << endl;
			exit(-1);
		}
		do
		{
			/* 	//遍历文件夹的图
			//判断是否有子目录
			if (FileInfo.attrib & _A_SUBDIR)
			{
			if ((strcmp(FileInfo.name, ".") != 0) && strcmp(FileInfo.name, "..") != 0)
			{
			string newPath = data_path + "\\" + FileInfo.name;
			}
			}
			*/
			string str_fname = data_path0 + "\\" + FileInfo.name;
			//图片，提取特征

			Mat tmp_img = imread(str_fname, CV_LOAD_IMAGE_GRAYSCALE);

			//放到训练集或测试集，写标签

			trian_count++;
			if (trian_count <= train_sample_count[ClassNum])
			{
				ExtrateImgFeature(tmp_img, _TrainFeature);
				std::vector<float> label;
				label.push_back(ClassNum);
				Mat label_mat(label);
				_TrainLabels.push_back(label_mat);
			}
			else
			{
				ExtrateImgFeature(tmp_img, _TestFeature);
				std::vector<float> label;
				label.push_back(ClassNum);
				Mat label_mat(label);
				_TestLabels.push_back(label_mat);
			}
		} while (_findnext(Handle, &FileInfo) == 0);
	}
	normlize();

	//PCA
	if (is_pca != 0)
	{
		int pcaRows = 900;

		CvMat pcaTrainData = _TrainFeature;

		CvMat* pcaMean = cvCreateMat(1, _TrainFeature.cols, CV_32FC1);
		CvMat* pcaEigVals = cvCreateMat(1, pcaRows, CV_32FC1);
		CvMat* pcaEigVecs = cvCreateMat(pcaRows, _TrainFeature.cols, CV_32FC1);
		cvCalcPCA(&pcaTrainData, pcaMean, pcaEigVals, pcaEigVecs, CV_PCA_DATA_AS_ROW);

		CvMat* _projcetTrainData = cvCreateMat(_TrainFeature.rows, pcaRows, CV_32FC1);
		cvProjectPCA(&pcaTrainData, pcaMean, pcaEigVecs, _projcetTrainData);
		_TrainFeature = _projcetTrainData;


		CvMat pcaTestData = _TestFeature;
		CvMat* _projcetTestData = cvCreateMat(_TestFeature.rows, pcaRows, CV_32FC1);
		cvProjectPCA(&pcaTestData, pcaMean, pcaEigVecs, _projcetTestData);
		_TestFeature = _projcetTestData;
	}

	FileStorage fs("..\\hog_pca_data.xml", FileStorage::WRITE);
	fs << "train_feature" << _TrainFeature;
	fs << "test_feature" << _TestFeature;
	fs << "train_labels" << _TrainLabels;
	fs << "test_labels" << _TestLabels;
	fs.release();
}


void Fer::LoadDataFromFile()
{
	FileStorage fs("..\\hog_pca_data.xml", FileStorage::READ);
	fs["train_feature"] >> _TrainFeature;
	fs["test_feature"] >> _TestFeature;
	fs["train_labels"] >> _TrainLabels;
	fs["test_labels"] >> _TestLabels;
	fs.release();
}

void Fer::normlize()
{

	//减均值除标准差
	Mat mean(1, _TrainFeature.cols, CV_32FC1), stddev(1, _TrainFeature.cols, CV_32FC1);

	for (int i = 0; i < _TrainFeature.cols; i++)
	{
		Scalar mean_sc, stddev_sc;
		meanStdDev(_TrainFeature.col(i), mean_sc, stddev_sc);

		mean.at<float>(0, i) = mean_sc.val[0];
		stddev.at<float>(0, i) = stddev_sc.val[0];
	}

	Mat norm_train_data;
	for (int i = 0; i < _TrainFeature.rows; i++)
	{
		for (int j = 0; j < _TrainFeature.cols; j++)
			_TrainFeature.at<float>(i, j) = _TrainFeature.at<float>(i, j) / 255;
		//_TrainFeature.at<float>(i, j) = (_TrainFeature.at<float>(i, j) - mean.at<float>(0, j)) / stddev.at<float>(0, j);
	}

	Mat norm_test_data;
	for (int i = 0; i < _TestFeature.rows; i++)
	{
		for (int j = 0; j < _TestFeature.cols; j++)
			_TestFeature.at<float>(i, j) = _TestFeature.at<float>(i, j) / 255;
		//_TestFeature.at<float>(i, j) = (_TestFeature.at<float>(i, j) - mean.at<float>(0, j)) / stddev.at<float>(0, j);
	}
}

void Fer::visualisze()
{

	FILE* fp = fopen("..//fer.txt", "a+");
	//统计每个列元素的最大值和最小值
	for (int i = 0; i < _TestFeature.cols; i++)
	{
		double maxv, minv;
		maxv = minv = _TestFeature.at<float>(0, i);

		for (int j = 0; j < _TestFeature.rows; j++)
		{
			if (maxv < _TestFeature.at<float>(j, i))
				maxv = _TestFeature.at<float>(j, i);
			if (minv > _TestFeature.at<float>(j, i))
				minv = _TestFeature.at<float>(j, i);
		}
		fprintf(fp, "%d cols: max : %.2f ,min : %.2f\n", i, maxv, minv);
	}
	fclose(fp);
}

int main()
{
	Fer f;
	for (int c = 0.1; c < 1; c += 0.1)
		f.run(c);
	return 0;
}