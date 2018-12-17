#include "LabArm.h"
#include <fstream>
#include <iostream>

//OpenCV
#include <opencv2/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <string>

using namespace cv;
using namespace cv::ml;

int main()
{
	//LabArm declaration:
	LabArm arm;
	//LabArm initialisation to mode 3 (Position Control)
	arm.MotorsInit(3);
	//arm.TorqueON();
	arm.GripperON();
		
	arm.motor5.SetOperatingMode(5);
	arm.motor5.SetGoalCurrent(500);
	
	arm.motor5.PrintOperatingMode();
	
	float WeightPosition5[6] = {180, 90, 180, 90, 270, 180};
	float PositionBetween[6] = {180, 110, 190, 130,230, 190};
	float WeightPosition2[6] = {270, 90, 270, 90, 180, 280};
	arm.StandBy();
	//arm.Goto(WeightPosition5, 1, 40, 15);
	
	printf("Loading the model...\n");
	Ptr<SVM> svm = SVM::load("../svm_v1_linear");
	printf("model loaded\n");
	
	arm.motor5.SetProfile(0, 0);
	arm.motor5.PrintPID();
	arm.motor5.SetPID(1200, 0, 0);	
	arm.motor5.PrintPID();
	
	printf("\n\n##############\n");
	printf("Press Enter to tar the gripper \n");
	std::cin.ignore();
	float weightCorrection=arm.Tar();
	printf("The weight of the gripper: %f\n",weightCorrection);
	int retar=0;
	while(1)
	{
		
		printf("\n\n##############\n");
		printf("Press Enter to close the gripper current goal = 90mA \n");
		std::cin.ignore();
		arm.gripper.SetGoalCurrent(90);
		sleep(0.5);
		arm.GripperClose();
		sleep(1);
		
		printf("Press Enter to go to Start the blind detection\n");
		std::cin.ignore();
		//Get features:
		float features[1][3];
		arm.GetFeatures(features, weightCorrection);
		Mat expSample(1, 3, CV_32FC1, features);
		std::cout<<std::endl<<expSample<<std::endl<<std::endl;
		//Classification:
		float pred=svm->predict(expSample);
		printf("\n\nPrediction: %f\n", pred);
		
		//Prediction processing
		String imageName;
		String labelWind;
		switch((int)pred)
		{
			case 1: imageName = "../Images/stone.jpeg";
				labelWind = "This is a stone!";
				printf("This is a stone.\n");
				break;
			case 2: imageName = "../Images/metal.jpeg";
				labelWind = "This is a metal piece!";
				printf("This is a metal piece.\n");
				break;
			case 3: imageName = "../Images/chestnuts.jpeg";
				labelWind = "This is a chestnuts!";
				printf("This is a chestnuts.\n");
				break;
			case 4: imageName = "../Images/foam.jpeg";
				labelWind = "This is a foam piece!";
				printf("This is a foam piece\n");
				break;
		}
		Mat image;
		image = imread( imageName, IMREAD_COLOR);
		if(image.empty())
		{
			printf("No image to open...\n");
		}
		
		namedWindow(labelWind, WINDOW_NORMAL);
		resizeWindow(labelWind, 300, 300);
		moveWindow(labelWind, 200,200);
		imshow(labelWind, image);
		waitKey(0);

		printf("Press Enter to grab another object\n");
		std::cin.ignore();
		arm.GripperOpen();
		arm.StandBy();
		sleep(1);
		retar++;
		if(retar==4)
		{
			printf("Press enter to recalibrate the weight tar\n");
			std::cin.ignore();
			weightCorrection=arm.Tar();
			retar=0;
		}
	}
	
	return(0);
}
