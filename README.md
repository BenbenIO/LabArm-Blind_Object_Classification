# LabArm - Blind Object Classification
This project focus on object identification based on the LabArm feedback only (no camera). The idea is to provide additional features to the camera in order to confirm and/or add more information about the grabbed object. This work was implemented as a normal function in the LabArm library but I separated into two different projects to let the LabArm-API simple and minimalist [HERE](https://github.com/BenbenIO/LabArm-Cpp-API). 
To get a better understanding of the project, please have a look at:
* [proof of concept video](https://youtu.be/eXRplSR5dM0)

<p align="center">
  <img src="/Images/armpicture.PNG" width="300">
</p>

### The challenge is to make the difference between a chesnut-stone, and plastic-glass bottle.
If you are willing to use this project, I would like to know on what kind of object you planned to use it :)

# Install && Dependencies
The programme depend on the __dynamixel_sdk__ library. Installation information can be found on their [github](https://github.com/ROBOTIS-GIT/DynamixelSDK). If you want to use a raspberry Pi please build and intall the SingleBoard Computer version (linux_sbc). For Joystick control, we based our function on [A minimal C++ object-oriented API onto joystick devices under Linux](https://github.com/drewnoakes/joystick), but the library is available on this repository.
<br/> Finally, you need to install __OpenCV__ (3.2.0) in order to use the SVM library.
<br/> Once the install is done clone this repository, cd into the make_run directory.
<br/> Make the MakeFile, an run the code ./exampleArm
<br/> You can add other library to the project by adding: __SOURCES += yourcode.cpp__ in the MakeFile.

# Principle
For a precise description, please read the [PDF file](https://github.com/BenbenIO/LabArm-Blind_Object_Classification/blob/master/Blind%20Object%20Recognition%20with%20LabArm.pdf), which describe all measurment principle and show classification proof of concept done in python (sklearn). I then implemented the SVM on C++ and integrate the solution into this project.

# Current Performances & Ongoing research
The accuracy of the function is worsen due to a change in the configuration between the implementation and the dataset creation. Another limitation is the instabillity of the weighting function (empty weight is not a constant). 
<br/> To solve this problematics, I am currently working on:</br>
* Collecting a new dataset to retrain the SVM.
* Changing the weight principle by incremmenting the goal current.

# Example Program
<br/> The following program will get all the features needed for the classification, and display a picture of the grabbed object. </br>

```c

LabArm arm;
arm.MotorsInit(3);
arm.GripperON(); 

arm.motor5.SetOperatingMode(5);
arm.motor5.SetGoalCurrent(500);
arm.motor5.PrintOperatingMode();

float WeightPosition5[6] = {180, 90, 180, 90, 270, 180};
arm.StandBy();

printf("Loading the model...\n");
Ptr<SVM> svm = SVM::load("../svm_v1_linear");
printf("model loaded\n");

arm.motor5.SetProfile(0, 0);
arm.motor5.SetPID(1200, 0, 0);	
arm.motor5.PrintPID();

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
   ```
