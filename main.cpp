#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv; 
using namespace std;

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    cout << tensor << endl;
    
    Mat image = imread("../../../../image.png",  // Path can be changed according to the location of the image
                       IMREAD_COLOR);
    // Error Handling 
    if (image.empty()) { 
        cout << "Image File "
             << "Not Found" << endl; 
  
        // wait for any key press 
        cin.get(); 
        return -1; 
    } 
  
    // Show Image inside a window with 
    // the name provided 
    imshow("Window Name", image); 
    
    cout << "Image Size" << image.size() << endl;
    // Wait for any keystroke 
    waitKey(0); 
    return 0; 


}
