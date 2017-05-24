#pragma once
#ifndef BAGOFWORDS_H
#define BAGOFWORDS_H
#include <string>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;

using namespace cv;

#define UNDEFINED -1
#define DEFAULT_CLUSTER_COUNT 300
//#define DEFAULT_FEATURE_COUNT 300

static const std::string SIFT_TYPE = "SIFT";
static const std::string SURF_TYPE = "SURF";

class BagOfWords {
public:
	static BagOfWords* Instance();
	// this method is a mirror of GetInstance
	static void ResetInstance()
	{
		delete s_instance; // REM : it works even if the pointer is NULL (does nothing then)
		s_instance = NULL; // so GetInstance will still work.
	}

	// delete copy and move constructors and assign operators
	BagOfWords(BagOfWords const&) = delete;
	BagOfWords(BagOfWords&&) = delete;
	BagOfWords& operator=(BagOfWords const&) = delete;
	BagOfWords& operator=(BagOfWords &&) = delete;

	
	BagOfWords* setFeature2D(const std::string type = "SUFR", int minHessian = 300, int nOctaves = 4, int nOctaveLayers = 3, bool extended = false, bool upright = false);
	Ptr<FeatureDetector> getFeature2D();
	Ptr<DescriptorExtractor> getDescriptorExtractor();


	BagOfWords* setBagOfWordsTrainer(int clusterCount ,int maxCount,  int attempts);
	Ptr<BOWKMeansTrainer> getBowTrainer();

	BagOfWords* setDescriptorMatcher(const std::string& type = "FlannBased");
	Ptr<DescriptorMatcher> getDescriptorMatcher();
	Ptr<BOWImgDescriptorExtractor> getBOWImageDescriptorExtractor();

	BagOfWords* setVocabularyStorage(const std::string& storagePath);
	std::string getVocabularyStorage();

	BagOfWords* setSVMStorage(const std::string& storagePath);
	std::string getSVMStorage();

	BagOfWords* setLogStorage(const std::string& storagePath);
	std::string getLogStorage();

	BagOfWords* setMatrixStorage(const std::string& storagePath);
	std::string getMatrixStorage(const std::string& t, const std::string& ext);

private:
	static BagOfWords *s_instance;
	BagOfWords() {};
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<BOWKMeansTrainer> bowTrainer;
	Ptr<DescriptorMatcher> descriptorMatcher;
	Ptr<BOWImgDescriptorExtractor> bowDE;

	std::string type;
	int dictionarySize;
	int minHess;
	int nOctaves;
	int nOctaveLayers;
	bool extended;
	bool upright;
	std::string storagePath = "."; 
	std::string vocabPath = ".";
	std::string svmPath = ".";
	std::string logPath = ".";
};

#endif
