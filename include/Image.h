#ifndef IMAGE_H
#define IMAGE_H
#include <vector>
#include "BagOfWords.h"
#include <boost/filesystem.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace boost::filesystem;
using namespace cv;

class Image {
public:
	Image();
	Image(const path&);
	~Image();

	std::string getImagePath();
	Mat getImage();
	
	std::vector<KeyPoint> getKeyPoints();
	Mat getDescriptors();
	Mat getHistogram();
	void showKeyPointImage();

private:
	Mat image;
	path image_path;
	Mat getImageFromFile();
	std::vector<KeyPoint> keypoints;
	Mat descriptors;
	Mat histogram;
};

#endif 
