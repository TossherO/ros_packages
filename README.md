# ros_packages
This is my personal project which contains all the ros packages I have created for robotic intelligent perceptions or other functions.

## Get Started
### 1. Prerequisites
I tested the code on OS ubuntu 20.04, using ROS Noetic. Make sure you are using the same operating system with ROS Noetic installed and configured. Also, your device should support CUDA version 11.x or 12.x， and have installed the corresponding CODA Toolkit.

### 2. Python Environment
**Step 1.**  Create a conda environment and activate it.  
```shell
conda create -n ros_perception python=3.10 -y
conda activate ros_perception
```

**Step 2.**  Install PyTorch
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
If you want to install a different version of PyTorch, you'd better choose the 2.1 version of PyTorch and make sure it's consistent with your CUDA version.

**Step 3.**  Install MMEngine, MMCV, MMDetection and MMDetection3D using MIM
```shell
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmdet3d==1.4.0
```

**Step 4.**  Install spconv and flash-attn
```shell
pip install spconv-cu120==2.3.6
pip install flash-attn==2.6.3
```
You can visit https://github.com/traveller59/spconv to choose the suitable version of spconv for you.  
Installing flash-attn directly via pip command may get stuck in the wheel package download process. You can visit https://github.com/Dao-AILab/flash-attention to download the corresponding wheel package, and then install flash-attn locally by pip command.

**Step 5.**  Install ROS-related packages
```shell
pip install rospkg empy
```

**Step 6.** Install other packages
```
pip install filterpy
```

### 3. Create ROS Workspace
```shell
cd $(workdir)
mkdir src
catkin_make
git clone https://github.com/TossherO/ros_packages.git
mv ros_packages/* src/
rm -rf ros_packages
catkin_make
source ./devel/setup.bash
```

### 4. Test Perception
**Step 1.** Download the data for test  
Download the zip file [CODA.zip](https://drive.google.com/file/d/11Wh5mzo2Bo14wTI92GREahCn4vjet0yT/view?usp=sharing). Then create a folder named "data/CODA" in the "\$(workdir)/src" directory and unzip "CODA.zip" to the folder.

**Step 2.** Download the model  
Download the model file [detect_coda.pth](https://drive.google.com/file/d/1OGpNygCHm8TqhHIPy13FNmypG9BdK-h6/view?usp=sharing). Then create a folder named "ckpts" in the "\$(workdir)/src/detection" directory and put "detect_coda.pth" in the folder.

**Step 3.** Run the test script
```shell
# terminal1
roscore

# terminal2
rosrun perception_test test_detection.py

# terminal3
rosrun perception_test test_tracking.py

# terminal4
rosrun perception_test test_traj_pred.py

# terminal5
rosrun perception_test test_visual.py
```

**Step 4.** Publish the command to the detection node
```shell
# terminal6
rostopic pub -r 5 /perception_input std_msgs/String "data: 'n'"
```
At this time you can see that terminal2~4 output infer information, and a window display the trajectory prediction results.