HF_HOME=/root/.cache/huggingface
HF_HUB_PATH=${HF_HOME}/hub

BASE_SERVER=nlp.36.torch2-shm
rsync -avP $BASE_SERVER:$HF_HUB_PATH $HF_HOME
