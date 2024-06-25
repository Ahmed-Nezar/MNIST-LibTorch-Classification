# OpenCV-LibTorch-Setup

## Introduction
This repositry explains how to setup both OpenCv &amp; PyTorch for C++ on windows operating system.
By Following the Below instructions hopefully you will be able to sucessfully run both OpenCV &amp; PyTorch with C++ on vscode & windows OS.

## Prequesites
- Go to PyTorch & Download LibTorch Cuda 11.8 by clicking [here](https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-debug-2.3.1%2Bcu118.zip)
- Make sure that cuda 11.8 is installed correctly & if not installed [click here to download the installer](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) **(make sure to choose your correct windows version)**
- Go to OpenCV & Download the **4.6.0 relaease** for windows you can find it [here](https://opencv.org/releases/)
- **Make sure you have microsft visual studio compiler installed on your system**
- Install **C/C++ Extension** in vscode along with **CMake Extension**

## Adjusting Environment Variables
- After downloading Each OpenCV & PyTorch extract them & copy both files to **`C:\Program Files` in order for the rest of the installation to work.** 
- Add the following to the path of your enviroment variables:
    ```
    C:\Program Files\opencv\build\x64\vc15\bin
    C:\Program Files\opencv\build\x64\vc15\lib
    C:\Program Files\libtorch\bin
    C:\Program Files\libtorch\lib
    ```
## Creating CMake & preparing vscode Environment
- Create an empty directory or folder for your project & open vscode in this directory
- Click on `Ctrl+shift+p` to open command palette or open it from settings
- After opening command palette you should see something like this:
  
    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/db623c0c-6038-4690-8951-5a2f104a4506)

- Type `CMake: Quick Start` & press `Enter`.
- You will be asked to enter the project name as below -> write **`OpenCVLibTorchTest`**:

    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/e49576e5-efce-4e6f-a901-b345aa88e610)

- Follow The Images Below:

  1.

    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/0aa813b1-0ca2-4858-8e94-6c021184e885)


  2.
   
    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/9addfc4c-1c23-4f38-9509-24df65b264d3)

  3.
   
    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/64a0c88d-f54a-48c2-a2f7-0ca94ad1be70)

  4. 

    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/169cebd4-ac98-42e2-b325-e01047d523f4)

  5. 

    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/e5a8c731-1618-4f35-aeca-f18276dbf1c8)

  6. **Press `Enter`**

- Copy the `CmakeLists.txt` in the repository & paste it in your local `CMakeLists.txt`
- Copy the `main.cpp` in the repository & paste it in your `main.cpp`
- See the image below to see how to build on vscode:

    ![image](https://github.com/Ahmed-Nezar/OpenCV-LibTorch-Setup/assets/125882735/1d9cd488-f11f-4103-82be-8a53df588608)

- The program should build & compiled successfully
  
## Getting Help
If you encounter any problems or have questions while setting up or using this repository, feel free to open an issue.

## Author
- Ahmed Nezar


  
