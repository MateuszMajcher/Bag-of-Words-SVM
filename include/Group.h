#pragma once
#ifndef GROUP_H
#define GROUP_H
#include <string>
#include <vector>
#include <chrono>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include <boost/filesystem.hpp>
#include "Image.h"
#include "File.h"

using namespace boost::filesystem;
using namespace cv;

class Group {
public:
	Group(path p);
	~Group();

	std::string getPath();
	std::string getName();
	std::vector<Image>& getImages();
	unsigned trainBOW();
	void getHistogram(std::vector<Mat>& output);
	void trainGroupClassifier();
	void trainGroupSVMClassifier(int index, Mat& trainData, Mat& labels);
	void predictImage(int index, Mat& trainData, Mat& labels);
	Mat getGroupClasifier();

private:
	std::string pathClass;
	std::string name;
	std::vector<Image> images;
	Mat groupClasifier;
	
};

#endif
