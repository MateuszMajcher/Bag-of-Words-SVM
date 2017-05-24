#pragma once
#ifndef IMAGEGROUP_H
#define IMAGEGROUP_H
#include <string>
#include <chrono>
#include <fstream>
#include <opencv2/ml.hpp>
#include <boost/filesystem.hpp>
#include "Group.h"
#include "File.h"
#include <omp.h>
using namespace boost::filesystem;
using namespace cv::ml;

class ImageGroup {
public:
	ImageGroup(const std::string& path);
	~ImageGroup();

	void train();
	void trainClassifier();
	void trainSVMClassifier(double C, double gamma);
	Group getImageClass(Image image);
	//void testImage(const std::string& p);
	void predictImage();
private:
	path p_source;
	std::vector<Group> groups;
	Ptr<SVM> svm;
};

#endif
