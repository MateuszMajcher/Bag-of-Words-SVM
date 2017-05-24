#include "BagOfWords.h"
#include <iostream>

BagOfWords *BagOfWords::s_instance = 0;


BagOfWords* BagOfWords::Instance() {
	if (!s_instance)
		s_instance = new BagOfWords;
	return s_instance;
}




// feature detectors and descriptor extractors
BagOfWords* BagOfWords::setFeature2D(const std::string type, int minHessian, int nOctaves, int nOctaveLayers, bool extended, bool upright) {
	this->type = type;
	this->minHess = minHessian;
	this->nOctaves = nOctaves;
	this->nOctaveLayers = nOctaveLayers;
	this->extended = extended;
	this->upright = upright;


	cv::Ptr<cv::FeatureDetector> _detector;
	cv::Ptr<cv::DescriptorExtractor> _extractor;
	if (type.compare(SURF_TYPE) == 0) {
		_detector = SURF::create(minHessian, nOctaves, nOctaveLayers,extended, upright);
		_extractor = SURF::create(minHessian, nOctaves, nOctaveLayers, extended, upright);
		std::cout << " created surf" << std::endl;
	}
	else if (type.compare(SIFT_TYPE) == 0) {
		_detector = cv::xfeatures2d::SURF::create();
		_extractor = cv::xfeatures2d::SURF::create();
		std::cout << "sift" << std::endl;
	}
	this->detector = _detector;
	this->extractor = _extractor;

	return this;
}

Ptr<FeatureDetector> BagOfWords::getFeature2D() {
	return detector;
}

Ptr<DescriptorExtractor> BagOfWords::getDescriptorExtractor() {
	return extractor;
}

BagOfWords* BagOfWords::setBagOfWordsTrainer(int clusterCount,int maxCount,  int attempts) {
	this->dictionarySize = clusterCount;
	/*TermCriteria tc(CV_TERMCRIT_ITER, maxCount, 0.001);
	int retries = attempts;
	int flags = KMEANS_PP_CENTERS;*/
	//this->bowTrainer = new BOWKMeansTrainer(clusterCount, tc, retries, flags);
	this->bowTrainer = new BOWKMeansTrainer(clusterCount);
	return this;
}

Ptr<BOWKMeansTrainer> BagOfWords::getBowTrainer() {
	return bowTrainer;
}

BagOfWords* BagOfWords::setDescriptorMatcher(const std::string& type) {
	this->descriptorMatcher = DescriptorMatcher::create(type);
	return this;
}

Ptr<DescriptorMatcher> BagOfWords::getDescriptorMatcher() {
	return descriptorMatcher;
}

Ptr<BOWImgDescriptorExtractor> BagOfWords::getBOWImageDescriptorExtractor() {
	if (!bowDE)
		bowDE = new BOWImgDescriptorExtractor(extractor, descriptorMatcher);
	return bowDE;
}


BagOfWords* BagOfWords::setMatrixStorage(const std::string& storagePath) {
	this->storagePath = storagePath;
	return this;
}


std::string BagOfWords::getMatrixStorage(const std::string& t, const std::string& ext) {
	return storagePath  + "\\"+  t + std::string("_") +type + "_" + std::to_string(dictionarySize) 
		+ "_" + std::to_string(minHess)
		+  "_" + std::to_string(nOctaves)
		+ "_" + std::to_string(nOctaveLayers)
		+ "_" + std::to_string(extended)
		+ "_" + std::to_string(upright)
		+  "." + ext;
}

BagOfWords* BagOfWords::setVocabularyStorage(const std::string& storagePath) {
	this -> vocabPath = storagePath;
	return this;
}
std::string BagOfWords::getVocabularyStorage() {
	return vocabPath;
}

BagOfWords* BagOfWords::setSVMStorage(const std::string& storagePath) {
	this->svmPath = storagePath;
	return this;
}
std::string BagOfWords::getSVMStorage() {
	return svmPath;
}

BagOfWords* BagOfWords::setLogStorage(const std::string& storagePath) {
	this->logPath = storagePath;
	return this;
}
std::string BagOfWords::getLogStorage() {
	return logPath;
}
