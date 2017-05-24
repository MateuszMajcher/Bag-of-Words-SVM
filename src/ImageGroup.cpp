#include "ImageGroup.h"

ImageGroup::ImageGroup(const std::string& p) {
	this->p_source = p;

	//vektor klas
	std::vector<path> className;
	className = getDirList(p);
	printVector(className);

	for (path pathGroup : className) {
		std::cout << "Utworzenie grupy dla " << pathGroup.generic_string() << std::endl;
		Group g(pathGroup);
		groups.push_back(g);
	}

	std::cout << "Utworzono " << groups.size() << " grup" << std::endl;
	
}

ImageGroup::~ImageGroup() {}

void ImageGroup::train() {
	BagOfWords* train = BagOfWords::Instance();
	std::ofstream log(train->getLogStorage(), std::ios_base::app);
	Mat vocabulary;

	unsigned int total_descriptor_count = 0;

	auto start = std::chrono::steady_clock::now();
	for (Group g : groups) {
		unsigned group_descriptor_count = 0;
		std::cout << "Training BOW for group " << g.getName() << std::endl;
		group_descriptor_count = g.trainBOW();
		total_descriptor_count += group_descriptor_count;
		std::cout << "- Adding " << g.getName() << group_descriptor_count << " training descriptors." << std::endl;
	}
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;

	std::cout << "Total descriptor count = " << total_descriptor_count << std::endl;
	log << "Totaldescriptor," << train->getBowTrainer()->descriptorsCount() << "\n";

	std::cout << "read descrptor in " << std::chrono::duration <double, std::milli>(diff).count() / 1000 << " s" << std::endl;
	log << "ExtractTime," << std::chrono::duration <double, std::milli>(diff).count() / 1000 << "\n";
	
	std::cout << "Running trainer" << std::endl;


	vocabulary.create(0, 1, CV_32FC1);
	start = std::chrono::steady_clock::now();
	vocabulary = train->getBowTrainer()->cluster();
	end = std::chrono::steady_clock::now();
	diff = end - start;
	
	std::cout << "create vocabulary in " << std::chrono::duration <double, std::milli>(diff).count() / 1000 << " s" << std::endl;
	log << "createTime," << std::chrono::duration <double, std::milli>(diff).count() / 1000 << "\n";
	saveMatrix(train->getVocabularyStorage(), vocabulary, "vocabulary");
}

void ImageGroup::trainClassifier() {
	std::cout << "******************************************************" << std::endl;
	std::cout << "Train classifiers" << std::endl;
	for (Group g : groups) {
		g.trainGroupClassifier();
	}
}

void ImageGroup::trainSVMClassifier(double C, double gamma) {
	std::cout << "Set C "<< C << " set gamma: " << gamma <<std::endl;
	BagOfWords* train = BagOfWords::Instance();
	std::ofstream log(train->getLogStorage(), std::ios_base::app | std::ios_base::out);
	Mat vocab;
	
	readMatrix(train->getVocabularyStorage(), vocab, "vocabulary");
	train->getBOWImageDescriptorExtractor()->setVocabulary(vocab);

	std::cout << "Set vacabulary "<< train->getVocabularyStorage() << " size: " << vocab.rows << std::endl;
	std::cout << "extracting histograms in the form of BOW for each image " << std::endl;


	Mat labels(0, 1, CV_32FC1);
	Mat trainingData(0, vocab.rows, CV_32FC1);
	int k = 1;

	auto start = std::chrono::steady_clock::now();
	for (Group g : groups) {
		unsigned group_descriptor_count = 0;
		std::cout << "Training BOW for group " << g.getName() << std::endl;
		g.trainGroupSVMClassifier(k ,trainingData, labels);
		k++;
	}
	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;

	std::cout << "extract histogram in " << std::chrono::duration <double, std::milli>(diff).count() / 1000 << " s" << std::endl;
	log << "ExtractHistogramTime," << std::chrono::duration <double, std::milli>(diff).count() / 1000 << "\n";

	std::cout << "training data size " << trainingData.size() << std::endl;
	std::cout << "training data size " << trainingData.rows<<" "<<trainingData.cols << std::endl;
	std::cout << "labels size " << labels.size() << std::endl;

	//ParamGrid ParamGrid_C(pow(2.0,-5), pow(2.0,15), pow(2.0,2));
	//ParamGrid ParamGrid_gamma(pow(2.0,-15), pow(2.0,3), pow(2.0,2));

	svm = SVM::create();
	///svm->setType(SVM::C_SVC);
	///svm->setKernel(SVM::RBF);
	//svm->setGamma(gamma);
	//svm->setC(C);
	//svm->setGamma(0.50625000000000009);
	//svm->setC(312.50000000000000);
	///svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS , 1000, 0.000001));
	// Train the SVM with given parameters
	cv::Ptr<cv::ml::TrainData> td =
        cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, labels);




	printf("%s\n", "Training SVM classifier");
	start = std::chrono::steady_clock::now();
	//bool res = svm->train(trainingData, ROW_SAMPLE, labels);
	svm->trainAuto(td);
	end = std::chrono::steady_clock::now();
	diff = end - start;

	std::cout<<"Kernel type "<<svm->getKernelType()<<std::endl;
	std::cout<<"svm Type "<<svm->getType()<<std::endl;
	std::cout<<"C "<<svm->getC()<<std::endl;
	std::cout<<"Gamma "<<svm->getGamma()<<std::endl;
	std::cout << "Train svm in " << std::chrono::duration <double, std::milli>(diff).count() / 1000 << " s" << std::endl;
	log << "TrainSVMTime," << std::chrono::duration <double, std::milli>(diff).count() / 1000 << "\n";


	svm->save(train->getSVMStorage());
	std::cout << "Save predictor" << std::endl;
}


/*void ImageGroup::testImage(const std::string& p) {
	BagOfWords* train = BagOfWords::Instance();
	Mat vocabulary;

	readMatrix(train->getVocabularyStorage(), vocabulary, "vocabulary");
	std::cout << "Set vacabulary " << vocabulary.rows << std::endl;
	std::cout << "extracting histograms in the form of BOW for each image " << std::endl;

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, vocabulary.rows, CV_32FC1);


}*/

void ImageGroup::predictImage() {
	BagOfWords* train = BagOfWords::Instance();
	std::ofstream log(train->getLogStorage(), std::ios_base::app | std::ios_base::out);

	/*Load vocabulary*/
	Mat vocab;
	readMatrix(train->getVocabularyStorage(), vocab, "vocabulary");
	train->getBOWImageDescriptorExtractor()->setVocabulary(vocab);
	std::cout << "Set vacabulary " << train->getVocabularyStorage() << " size: " << vocab.rows << std::endl;
	/*Load SVM*/
	cv::Ptr<cv::ml::SVM> mSvm2;
	std::cout << "load svm classifer" << std::endl;
	mSvm2 = Algorithm::load<SVM>(train->getSVMStorage());

	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalData(0, 1500, CV_32FC1);
	Mat results(0, 1, CV_32FC1);
	int k = 1;

	int good = 0, bad = 0;
	int total = 0;

	auto start = std::chrono::steady_clock::now();
	for (Group g : groups) {
		for (Image img : g.getImages()) {
			total++;
			std::vector<KeyPoint> keypoint2;
			Mat bowDescriptor2;
			//std::cout << "read " << img.getImagePath() << img.getImage().cols << std::endl;
			BagOfWords::Instance()->getFeature2D()->detect(img.getImage(), keypoint2);
			BagOfWords::Instance()->getBOWImageDescriptorExtractor()->compute(img.getImage(), keypoint2, bowDescriptor2);
			evalData.push_back(bowDescriptor2);
			groundTruth.push_back((float)k);
			float response = mSvm2->predict(bowDescriptor2);
			//std::cout << "category:  " << k << " - "<<g.getName()<< std::endl;
			//std::cout << "predict " << response <<" - "<<groups[response-1].getName()<< std::endl;
			//if (k != response)
			std::cout << img.getImagePath()<< std::endl;
			log << k << ":" << response << ",";
			if (k == response) good++; else bad++;

			results.push_back(response);
		}
		k++;
	}

	auto end = std::chrono::steady_clock::now();
	auto diff = end - start;

	std::cout << "predict in " << std::chrono::duration <double, std::milli>(diff).count() / 1000 << " s" << std::endl;
	log << "predictTime," << std::chrono::duration <double, std::milli>(diff).count() / 1000 << "\n";

	
	std::cout << "good " << good << " bad: " << bad << std::endl;
	log<< "good " << good << " bad: " << bad << "\n";
	std::cout << "total " << total  << std::endl;


	//calculate the number of unmatched classes 
	double errorRate = (double)countNonZero(groundTruth - results) / evalData.rows;
	printf("%s%f", "Error rate is ", errorRate);
	log << "Error rate is "<< errorRate << "\n";
}


Group ImageGroup::getImageClass(Image image) {
	float bestFit = 0;
	int bestFitPos = -1;

#pragma omp parallel for 
	for (int i = 0; i < groups.size(); i++) {
		float currentFit = getHistogramIntersection(groups[i].getGroupClasifier(), image.getHistogram());
		std::cout << groups[i].getName() << " " << currentFit << std::endl;
#pragma omp critical
		{
			if (currentFit > bestFit) {
				bestFit = currentFit;
				bestFitPos = i;
			}
		}
	}
	return groups[bestFitPos];
}
