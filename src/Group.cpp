#include "Group.h"

Group::Group(path p) {

	this->name = p.filename().generic_string();
	this->pathClass = p.generic_string();

	std::vector<std::string> image_source;
	getAll(p, ".jpg", image_source);
	//dodanie zdjec
	for (std::string s : image_source)
		images.push_back(Image(s));
}

Group::~Group()
{
}

std::string Group::getPath() {
	return this->pathClass;
}

std::string Group::getName() {
	return this->name;
}

std::vector<Image>& Group::getImages() {
	return images;
}


unsigned Group::trainBOW() {
	unsigned descriptor_count = 0;
	Ptr<BOWKMeansTrainer> trainer = BagOfWords::Instance()->getBowTrainer();
	
#pragma omp parallel for shared(trainer, descriptor_count)
	for (int i = 0; i < (int)images.size(); i++) {
		auto start = std::chrono::steady_clock::now();
		Mat descriptors = images[i].getDescriptors();
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
#pragma omp critical
		{
			trainer->add(descriptors);
			descriptor_count += descriptors.rows;
			std::cout <<name<<" - "<< descriptors.rows <<" - "<<std::chrono::duration <double, std::milli>(diff).count()/1000 << " s" << std::endl;
		}
	}
	return descriptor_count;
}

void Group::trainGroupClassifier() {
	BagOfWords* prop = BagOfWords::Instance();
	std::cout <<"Train classifier for " <<name<< std::endl;
	if (!readMatrix(prop->getVocabularyStorage(), groupClasifier, name)) {
		std::vector<Mat> groupHistogram;
		getHistogram(groupHistogram);
		getMedianHistogram(groupHistogram, groupClasifier);
		saveMatrix(prop->getVocabularyStorage(), groupClasifier, name);
	}
}

void Group::trainGroupSVMClassifier(int index, Mat& trainData, Mat& labels) {
#pragma omp parallel for shared(output)
	for (int i = 0; i < (int)images.size(); i++) {
		
		Mat imageHistogram = images[i].getHistogram();
		std::cout <<images[i].getImagePath()<< "get Histogram "<<imageHistogram.cols<<std::endl;

#pragma omp critical
		{
			trainData.push_back(imageHistogram);
			labels.push_back(index);
		}
	}
}

void Group::predictImage(int index, Mat& trainData, Mat& labels) {
#pragma omp parallel for shared(output)
	for (int i = 0; i < (int)images.size(); i++) {

		Mat imageHistogram = images[i].getHistogram();
		std::cout << images[i].getImagePath() << "get Histogram " << imageHistogram.cols << std::endl;
		//printHistogram(imageHistogram);
#pragma omp critical
		{
			trainData.push_back(imageHistogram);
			labels.push_back(index);
		}
	}
}





/*Pobranie histogramów dla grupy*/
void Group::getHistogram(std::vector<Mat>& output) {
#pragma omp parallel for shared(output)
	for (int i = 0; i < (int)images.size(); i++) {
		std::cout << "get Histogram ";
		Mat imageHistogram = images[i].getHistogram();
		printHistogram(imageHistogram);

#pragma omp critical
		{
			output.push_back(imageHistogram);
		}
	}
}

Mat Group::getGroupClasifier()
{
	if (groupClasifier.empty()) trainGroupClassifier();
	return groupClasifier;
}