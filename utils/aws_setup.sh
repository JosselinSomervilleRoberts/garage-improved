# Save current path
pwd=`pwd`

# Create conda env and activate it
yes y | conda create -n garage_improved python==3.8
conda init bash
source ~/.bashrc
conda activate garage_improved

# Install tensorflow and pytorch
# This should get isntalled with garage
# yes y | pip install tensorflow
# yes y | conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# yes y | conda install tensorflow-estimator

# Install joss toolbox
yes y | pip install git+https://github.com/JosselinSomervilleRoberts/JossPythonToolbox.git

# Install Mujoco
cd /home/ubuntu
mkdir .mujoco
cd .mujoco
wget https://www.roboti.us/file/mjkey.txt # Key
wget https://www.roboti.us/download/mujoco200_linux.zip # Mujoco 200
yes y | sudo apt-get install unzip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm -r -f mujoco200_linux.zip
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz # Mujoco 210 (Not needed)
tar -xvf mujoco210-linux-x86_64.tar.gz
rm -r -f mujoco210-linux-x86_64.tar.gz
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco200/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
source ~/.bashrc
conda activate garage_improved
yes y | sudo apt-get install patchelf
yes y | sudo apt-get install libglew-dev
yes y | pip install mujoco
yes y | pip install scipy
yes y | sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
cd ..

# Install garage
yes y | pip install -e '.[mujoco,dm_control,tensorflow]'

# Install metaworld
yes y | sudo apt install swig # required for box2d (See https://github.com/openai/spinningup/issues/32)
yes y | pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
yes y | pip install 'gym[box2d]'
# Fix metaworld bug 'Maximum path length allowed by the benchmark has been exceeded'
cd $pwd
cp ./mujoco_env_file_fixed.py /opt/conda/envs/garage_improved/lib/python3.8/site-packages/metaworld/envs/mujoco/mujoco_env.py