#
# NVIDIA NGC コンテナを使う
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# 動作確認されているので、独自にコンテナをビルドするより手間が少ないです
#
FROM nvcr.io/nvidia/pytorch:22.02-py3

#
# 日本時間に設定する (ログを見やすくするため)
#
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev tzdata \
&&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

#
# プロジェクトに必要なパッケージをインストールする
#
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    mmcv-full==1.6.0

#
# requirements/runtime.txt をインストールする
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    matplotlib \
    numpy \
    pycocotools \
    six \
    terminaltables

#
# ClearML関連のパッケージ
#
RUN pip3 install --no-cache-dir --use-deprecated=legacy-resolver \
    clearml-agent \
    boto3
