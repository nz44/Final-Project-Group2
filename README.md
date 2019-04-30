# Final-Project-Group2
Spring 2019 Machine Learning II Final Project CNN Painting Classification
# If you do not want to download the dataset from Kaggle API
The resized folder all the images in the original dataset. It is included in the git repo. 
sudo unzip resized.zip
sudo chmod -R 777 /<resized folder path>
Then you can start running the code in the suggested sequence below (skip next section). 

# If you would like to download the dataset from Kaggle API
Please run the following command on the terminal
## Give permission to your home directory
sudo chmod -R 777 /home <or your home directory>

## Install Kaggle API
pip install --user kaggle
mkdir ~/.kaggle
(Generate Kaggle API token according to https://github.com/Kaggle/kaggle-api and place it in the above folder)
chmod 600 ~/.kaggle/kaggle.json

## Create your working directory
mkdir /home/<your working directory>/
cd  /home/<your working directory>/

## Download Kaggle Dataset and Unzip it
kaggle datasets download -d ikarus777/best-artworks-of-all-time
sudo unzip best-artworks-of-all-time.zip
sudo unzip resized.zip

## Give permission to extracted resized folder
sudo chmod -R 777 /<resized folder path>

## Set the display
export DISPLAY=localhost:10.0

# Run the code in suggested sequence because they match with the report
0_load_and_proprocess.py
1_cnn_model_learningrate.py
1_cnn_model_minibatch.py
2_cnn_model_2.py
22_cnn_model_2.py
3_cnn_model_3.py
4_cnn_model_4.py
5_cnn_model_5.py




