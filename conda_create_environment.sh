conda create -y -n cross_task python==3.9
conda activate cross_task
conda install -y numba
pip install librosa
pip install ffmpeg
pip install pandas==1.4.4
pip install h5py
pip install  scipy
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch==1.13.1
pip install pytorch-lightning==1.5.0
pip install asteroid==0.4.1
pip install psds_eval
pip install sed_eval==0.2.1
pip install sed-scores-eval==0.0.0
pip install thop
pip install torchmetrics==0.7.3
pip install codecarbon