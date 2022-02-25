# Install app from local directory
conda install -y "pytorch==1.9.1" "torchvision==0.10.1" -c pytorch
conda install -y scipy "scikit-learn==0.24.2" "scikit-image==0.18.3" -c conda-forge

pip install /local-app
