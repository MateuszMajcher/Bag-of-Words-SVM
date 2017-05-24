#include "Image.h"
#include <iostream>
Image::Image() {}

Image::Image(const path& path) {
	this->image_path = path;
	getImage();
}

Image::~Image() {}

Mat Image::getImage() {
	return (this->image.empty()) ? this->image = getImageFromFile() : this->image;
}

Mat Image::getImageFromFile() {
	Mat image = imread(this->image_path.generic_string(), IMREAD_GRAYSCALE);
	if (!image.data)
	{
		std::cout << " --(!) Error reading images " << std::endl;
	}
	return image;
}

std::string Image::getImagePath() {
	return image_path.generic_string();
}

std::vector<KeyPoint> Image::getKeyPoints() {
	if (keypoints.empty())
		BagOfWords::Instance()->getFeature2D()->detect(this->image, keypoints);
	return this->keypoints;
}

Mat Image::getDescriptors() {
	if (getKeyPoints().empty()) getKeyPoints();
	BagOfWords::Instance()->getDescriptorExtractor()->compute(this->image, keypoints, descriptors);
	return descriptors;
}

Mat Image::getHistogram() {
	if (getKeyPoints().empty()) getKeyPoints();
	BagOfWords::Instance()->getBOWImageDescriptorExtractor()->compute(this->image, keypoints, histogram);
	return histogram;
}

void Image::showKeyPointImage() {
	if (getKeyPoints().empty()) getKeyPoints();
	Mat img_keypoints;
	drawKeypoints(this->image, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("image", img_keypoints);
}