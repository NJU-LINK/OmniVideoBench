set -ex
source /mnt/shared-storage-user/colab-share/liujiaheng/workspace/caoruili/omni-videos-lcr/code/env/minicpm/bin/activate
cd /mnt/shared-storage-user/colab-share/liujiaheng/workspace/caoruili/omni-videos-lcr/code/omni-bench/model
python minicpm-o_eval.py >> test_minicpmo-0916.log