//Load the dataset and evaluate the SVM:
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using namespace cv::ml;
using namespace std;

int main(int, char**)
{
	//Loading DataSet
	ifstream file ("SizeToughWeight.csv");
	if(file.is_open())
	{
		printf("File Opened");
	}
	else
	{
		printf("Error while opening the dataset...");
	}


	string line="";
	float dataset[28][4];
	int i=0, c=0;

	while(getline(file, line))
	{
		cout<<line<<"\n";
		//Parsing the line:
		stringstream stream(line);
		string cell;
		while(getline(stream,cell, ','))
		{
			dataset[i][c]=atof(cell.c_str());  //Convert float to string
			c++;
		}
		c=0;
		i++;
	}
	file.close();


	//Setting up the training data (add normalization?)
	int label[28];
	float features[28][3];
	for(int i=0;i<27; i++)
	{
		label[i]=dataset[i][0];
		features[i][0]=dataset[i][1];
		features[i][1]=dataset[i][2];
		features[i][2]=dataset[i][3];
	}
	Mat featureMat(28, 3, CV_32FC1, features);
	Mat labelMat(28, 1, CV_32SC1, label);


	// Set and train SVM
	printf("Loading the model...");
	Ptr<SVM> svm = SVM::load("svm_v1_linear");
	printf("model loaded\n");
	
	//test:
	for(int t=0; t<27; t++)
	{
		Mat testsample = featureMat.row(t);
		float pred=svm->predict(testsample);
		cout<<"n"<<t<<" True: "<<labelMat.row(t)<<" Pred: "<<pred<<endl;
	}
	//Try on random:
	float exp[1][3]={10, 18,2.3}; //chestnut
	Mat expSample(1, 3, CV_32FC1, exp);
	float pred=svm->predict(expSample);
	printf("\n\nOther prediction : chestnut? %f\n", pred);
	return(0);
}
