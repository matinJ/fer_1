#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;

class Fer
{
public :
	Fer()
	{
		HandleFeatureAndLabels(0);
		LoadDataFromFile();
		visualisze();
	}
	~Fer(){}
	double run(double c);
private:
	void HandleFeatureAndLabels(int ispca);
	void LoadDataFromFile();
	void normlize();
	void ExtrateImgFeature(
		Mat& imgs,
		Mat& featuresMat,
		Size dsize = Size(48, 48),
		Size winSize = Size(48, 48),
		Size blockSize = Size(8, 8),
		Size blockStride = Size(2, 2),
		Size cellSize = Size(4, 4),
		int nbins = 9,
		int derivAperture = 1,
		double winSigma = -1,
		int histogramNormType = HOGDescriptor::L2Hys,
		double L2HysThreshold = 0.2,
		bool gammaCorrection = true,
		int nlevels = HOGDescriptor::DEFAULT_NLEVELS);
	void visualisze();
	Mat _TrainFeature;
	Mat _TestFeature;
	Mat _TrainLabels;
	Mat _TestLabels;
};