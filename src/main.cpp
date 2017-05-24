#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include "File.h"
#include "Image.h"
#include "ImageGroup.h"


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;




int main(int argc, const char** argv)
{
cout<<argc<<endl;
	if (argc != 13)
	{
		cout << "No arguments.You should run this program in terminal with several arguments." << endl;
		cout << "Use [matchFile] [countFile] [witchStat]" << endl;
		exit(1);
	}

	std::string input_dir(argv[1]);
	std::string vocab_file(argv[2]);
	std::string svm_file(argv[3]);
	int cluster = atoi(argv[4]);
	int minHessian = atoi(argv[5]);
	int nOctaves = atoi(argv[6]);
	int nOctaveLayers = atoi(argv[7]);
	bool extended = atoi(argv[8]);
	bool upright = atoi(argv[9]);
	int maxCount = atoi(argv[10]);
	int attempts = atoi(argv[11]);
	std::string log(argv[12]);

	std::cout << "test_dir " << input_dir << std::endl;
	std::cout << "vocab_file " << vocab_file << std::endl;
	std::cout << "svm_file " << svm_file << std::endl;
	std::cout << "cluster " << cluster << std::endl;
	std::cout << "minHessian " << minHessian << std::endl;
	std::cout << "nOctaves " << nOctaves << std::endl;
	std::cout << "nOctaveLayers " << nOctaveLayers << std::endl;
	std::cout << "extended " << extended << std::endl;
	std::cout << "upright " << upright << std::endl;
	std::cout << "maxCount " << maxCount << std::endl;
	std::cout << "attempts " << attempts << std::endl;
	std::cout << "log " << log << std::endl;

	BagOfWords::Instance()->setFeature2D("SURF", minHessian, nOctaves, nOctaveLayers, extended, upright)
		->setBagOfWordsTrainer(cluster, maxCount, attempts)
		->setDescriptorMatcher("FlannBased")
		->setVocabularyStorage(vocab_file)
		->setSVMStorage(svm_file)
		->setLogStorage(log);

	ImageGroup testImages(input_dir);
	testImages.predictImage();

	/*if (argc != 15)
	{
		cout << "No arguments.You should run this program in terminal with several arguments." << endl;
		cout << "Use [matchFile] [countFile] [witchStat]" << endl;
		exit(1);
	}

	std::string input_dir(argv[1]);
	std::string vocab_file(argv[2]);
	std::string svm_file(argv[3]);
	int cluster = atoi(argv[4]);
	int minHessian = atoi(argv[5]);
	int nOctaves = atoi(argv[6]);
	int nOctaveLayers = atoi(argv[7]);
	bool extended = atoi(argv[8]);
	bool upright = atoi(argv[9]);
	int maxCount = atoi(argv[10]);
	int attempts = atoi(argv[11]);
	double c = stod(argv[12]);
	double gamma = stod(argv[13]);
	std::string log(argv[14]);

	std::cout << "input_dir " << input_dir << std::endl;
	std::cout << "vocab_file " << vocab_file << std::endl;
	std::cout << "svm_file " << svm_file << std::endl;
	std::cout << "cluster " << cluster << std::endl;
	std::cout << "minHessian " << minHessian << std::endl;
	std::cout << "nOctaves " << nOctaves << std::endl;
	std::cout << "nOctaveLayers " << nOctaveLayers << std::endl;
	std::cout << "extended " << extended << std::endl;
	std::cout << "upright " << upright << std::endl;
	std::cout << "c " << c << std::endl;
	std::cout << "gamma " << gamma << std::endl;
	std::cout << "log " << log << std::endl;

	ImageGroup imageGroup(input_dir);

	BagOfWords::Instance()->setFeature2D("SURF", minHessian, nOctaves, nOctaveLayers, extended, upright)
		->setBagOfWordsTrainer(cluster, maxCount, attempts)
		->setDescriptorMatcher("FlannBased")
		->setVocabularyStorage(vocab_file)
		->setSVMStorage(svm_file)
		->setLogStorage(log);


	imageGroup.trainSVMClassifier(c, gamma);*/

	/*if (argc != 12)
	{
		cout << "No arguments.You should run this program in terminal with several arguments." << endl;
		cout << "Use [matchFile] [countFile] [witchStat]" << endl;
		exit(1);
	}

	std::string input_dir(argv[1]);
	std::string vocab(argv[2]);
	int cluster = atoi(argv[3]);
	int minHessian = atoi(argv[4]);
	int nOctaves = atoi(argv[5]);
	int nOctaveLayers = atoi(argv[6]);
	bool extended = atoi(argv[7]);
	bool upright = atoi(argv[8]);
	int maxCount = atoi(argv[9]);
	int attempts = atoi(argv[10]);
	std::string log(argv[11]);

	std::cout << "input_dir " << input_dir << std::endl;
	std::cout << "vocab " << vocab << std::endl;
	std::cout << "cluster " << cluster << std::endl;
	std::cout << "minHessian " << minHessian << std::endl;
	std::cout << "nOctaves " << nOctaves << std::endl;
	std::cout << "nOctaveLayers " << nOctaveLayers << std::endl;
	std::cout << "extended " << extended << std::endl;
	std::cout << "upright " << upright << std::endl;
	std::cout << "maxCount " << maxCount << std::endl;
	std::cout << "attempts " << attempts << std::endl;
	std::cout << "log " << log << std::endl;


	ImageGroup imageGroup(input_dir);

	BagOfWords::Instance()->setFeature2D("SURF", minHessian, nOctaves, nOctaveLayers, extended, upright)
		->setBagOfWordsTrainer(cluster, maxCount, attempts)
		->setDescriptorMatcher("FlannBased")
		->setVocabularyStorage(vocab)
		->setLogStorage(log);

//	printHistogram(i.getHistogram());
	//trenowanie slownika
	imageGroup.train();
	//trenowanie klasyfikatora
	//imageGroup.trainSVMClassifier();
	//std::cout << "Classifiers trained" << std::endl;
	//imageGroup.testImage(test_dir);*/

	
	return 0;
}
