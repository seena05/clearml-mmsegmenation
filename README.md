# ClearMLで外部プロジェクトを取り込んで実験を管理する方法

ClearMLを使ってリモートマシン上で学習を実行するには以下の準備が必要です

1. 実験対象のレポジトリーの追加 (リモートでコードを参照するため)
2. clearml-data を使ったデータセットの管理 (リモートでデータセットを参照するため)
3. Dockerコンテナを作る (リモートで実行する環境を用意するため)
4. `clearml-task` 用のコードを用意する (分散実行したりデータを用意するため)
5. トレーニングをログする (GUI上で確認するため)

上記の準備ができたら、`clearml-task` を利用してリモートでタスクを実行できるようになります

## 1.実験対象のレポジトリーの追加

[Git Subtree](https://docs.github.com/en/get-started/using-git/about-git-subtree-merges) を利用して、コードの変更を管理できるようにします。

単純にソースコードをコピーして入れる場合、参照先の変更について追従できなくなりますが、この方法であれば`git`の機能により変更をマージすることができます。

参照先のプロジェクトを [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) を例に説明します。

### 01. リモートURLを追加する

```
$ git remote add -f mmdetection https://github.com/open-mmlab/mmdetection.git
Updating mmdetection
remote: Enumerating objects: 24328, done.
remote: Total 24328 (delta 0), reused 0 (delta 0), pack-reused 24328
Receiving objects: 100% (24328/24328), 37.48 MiB | 12.36 MiB/s, done.
Resolving deltas: 100% (17017/17017), done.
From https://github.com/open-mmlab/mmdetection
 * [new branch]        circleci-project-setup -> mmdetection/circleci-project-setup
 * [new branch]        dev                    -> mmdetection/dev
 * [new branch]        master                 -> mmdetection/master
 * [new branch]        yolov4                 -> mmdetection/yolov4
 * [new tag]           v2.24.1                -> v2.24.1
 * [new tag]           v0.6.0                 -> v0.6.0
 * [new tag]           v0.6rc0                -> v0.6rc0
 (...snip...)
```

### 02. ローカルレポジトリにマージする

```
$ git merge -s ours --no-commit --allow-unrelated-histories mmdetection/master
Automatic merge went well; stopped before committing as requested
```

※ 最近はmaster/mainのブランチ名が混在しているので都度確認して下さい

#### 特定のバージョン(タグ)を取り出す場合

```
$ git merge -s ours --no-commit --allow-unrelated-histories v2.24.1
```

### 03. サブディレクトリに追加する

```
$ git read-tree --prefix=mmdetection/ -u mmdetection/master
```

#### 特定のバージョン(タグ)を取り出す場合

```
$ git read-tree --prefix=mmdetection/ -u v2.24.1
```

### 04. コミットする

```
$ git commit -m 'subtree merge'
```

### 05. 変更を取り入れる (オプション)

master/mainを取り入れている場合は、以下の方法で差分を取り入れることができます

```
$ git pull -s subtree mmdetection master
```

## 2. clearml-data を使ったデータセットの管理

[clearml-data](https://clear.ml/docs/latest/docs/clearml_data/clearml_data) を使って、データセットをリモートから参照できるようにします

通常のS3やNFSを使うのと比較してキャッシュ機能とClearMLの純正の機能であることからコード中で参照しやすくて便利です

### 01. データセットを準備する

できればCOCO形式に変換するなどすると使い易いでしょう

COCO2017を `data/coco` に配置した例を示します

```
$ cd data
$ ls coco
annotations  train2017  val2017
```

### 02. データセットプロジェクトを作る

`clearml-data create` によりデータセット用のプロジェクトをClearML上に作成できます

```
$ clearml-data create --project public-dataset --name coco2017
clearml-data - Dataset Management & Versioning CLI
Creating a new dataset:
New dataset created id=227750fe4a824e2e9cfe214d6d27b250
```

`id=25d3c555841a4e5583861bc74d44b0de` が出力されますが、以後このIDでデータセットの定義を行います

### 03. データを追加する

`clearml-data add` により一括でデータを追加できます

```
$ clearml-data add --id 227750fe4a824e2e9cfe214d6d27b250 --files coco
clearml-data - Dataset Management & Versioning CLI
Adding files/folder to dataset id 227750fe4a824e2e9cfe214d6d27b250
Generating SHA2 hash for 2533 files
Hash generation completed
2533 files added
```

※ 説明用にデータ数を削減しています

### 04. データをアップロードする

アップロードする前に `~/clearml.conf` を編集して、`storage.dev.ai-ms.com` にアクセスできるようにしておきます

```
sdk {
    aws {
        s3 {
            credentials: [
                {
                     host: "storage.dev.ai-ms.com:9000"
                     bucket: "datasets"
                     key: "HW50AU3UR84S14TZHZV2"
                     secret: "D5RjOinInevAT9m7ioJVn6GItJOKCjsIwI3pVnGt"
                     multipart: false
                     secure: false
                }
            ]
        }
    }
}
```

`clearml-data upload` により一括でデータを追加できます

```
$ clearml-data upload --id 227750fe4a824e2e9cfe214d6d27b250 --storage s3://storage.dev.ai-ms.com:9000/datasets/
clearml-data - Dataset Management & Versioning CLI
uploading local files to dataset id 227750fe4a824e2e9cfe214d6d27b250
Compressing local files, chunk 1 [remaining 2533 files]
File compression completed: total size 412.73 MB, 1 chunked stored (average size 412.73 MB)
Uploading compressed dataset changes 1/1 (2533 files 412.73 MB) to s3://storage.dev.ai-ms.com:9000/datasets
2022-05-07 17:24:13,603 - clearml.storage - INFO - Uploading: 5.00MB / 393.61MB @ 65.64MBs from /tmp/dataset.227750fe4a824e2e9cfe214d6d27b250.zip
2022-05-07 17:24:13,681 - clearml.storage - INFO - Uploading: 10.00MB / 393.61MB @ 64.08MBs from /tmp/dataset.227750fe4a824e2e9cfe214d6d27b250.zip
(...snip...)
```

### 05. データを固定する

`clearml-data close` によりデータセットを固定することができます。

```
$ clearml-data close --id 227750fe4a824e2e9cfe214d6d27b250
clearml-data - Dataset Management & Versioning CLI
Finalizing dataset id 227750fe4a824e2e9cfe214d6d27b250
2022-05-07 17:28:53,270 - clearml.Task - INFO - Waiting to finish uploads
2022-05-07 17:28:53,338 - clearml.Task - INFO - Finished uploading
Dataset closed and finalized
```

これでデータセットが利用可能になりました。
APIを通じて誤ってデータを変更することが無くなります。
新しいデータセットを作る場合は、継承して別のデータセットとして作成します。

※ `--id` を `create` 以降の全てのコマンドに記載しましたが、`close` まで状態として保持されるので一連の作業を連続して行う場合は省略可能です

```
$ pip install clearml boto3
$ cd mmdetection
$ clearml-data get --id d6507bf2a2d54f66952a8cdd540f3a09 --copy data/coco
```

## 3. Dockerコンテナを作る

手元の環境でDockerを利用して単独で学習できる環境を作成し、その環境をリモートで実行できるようにする方法を説明します

### 01. ローカルでの環境作成

`Dockerfile`、`docker-compose.yaml` を作成して、実験環境のコンテナを作成します

#### Dockerfileの注意点

後から`clearml`の機能を使ってパッケージを入れることもできますが、時間が掛かるものは事前に入れておくと実験の開始が早くなります

#### コンテナのビルド

```
$ docker-compose build
Building clearml_example_mmdetection
Sending build context to Docker daemon  83.86MB
Step 1/4 : FROM nvcr.io/nvidia/pytorch:22.02-py3
 ---> e680c1c45ecd
Step 2/4 : RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y     libgl1-mesa-dev tzdata &&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime &&  apt-get clean &&  rm -rf /var/lib/apt/lists/*
 ---> Using cache
(...snip...)
```

#### コンテナの立ち上げ

```
$ docker-compose up
Creating network "clearml-example-mmdetection_default" with the default driver
Creating clearml-example-mmdetection_clearml_example_mmdetection_1 ... done
Attaching to clearml-example-mmdetection_clearml_example_mmdetection_1
clearml_example_mmdetection_1  |
clearml_example_mmdetection_1  | =============
clearml_example_mmdetection_1  | == PyTorch ==
clearml_example_mmdetection_1  | =============
clearml_example_mmdetection_1  |
clearml_example_mmdetection_1  | NVIDIA Release 22.02 (build 32255746)
clearml_example_mmdetection_1  | PyTorch Version 1.11.0a0+17540c5
(...snip...)
```

#### コンテナに入る

別のターミナルから以下を実行します

※ `clearml_example_mmdetection` は `docker-compose.yaml` に記載されているサービス名です

```
$ docker-compose exec clearml_example_mmdetection /bin/bash
root@0711357be4e0:/workspace#
```

#### 動作確認

テストデータを `mmdetection/data/` に配置してからローカルの `mmdet` をインストールして実行します

(※ 事前に`mmdetection/data/coco` にCOCO2017データセットを配置していると仮定します)

```
root@0711357be4e0:/workspace# cd mmdetection
root@0711357be4e0:/workspace# pip install -e .
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Obtaining file:///workspace/mmdetection
Requirement already satisfied: matplotlib in /opt/conda/lib/python3.8/site-packages (from mmdet==2.24.1) (3.5.1)
Requirement already satisfied: numpy in /opt/conda/lib/python3.8/site-packages (from mmdet==2.24.1) (1.22.2)
Requirement already satisfied: pycocotools in /opt/conda/lib/python3.8/site-packages (from mmdet==2.24.1) (2.0+nv0.6.0)
Requirement already satisfied: six in /opt/conda/lib/python3.8/site-packages (from mmdet==2.24.1) (1.16.0)
(...snip...)

root@0711357be4e0:/workspace# python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
NOTE! Installing ujson may make loading annotations faster.
/workspace/mmdetection/mmdet/utils/setup_env.py:38: UserWarning: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  warnings.warn(
/workspace/mmdetection/mmdet/utils/setup_env.py:48: UserWarning: Setting MKL_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
  warnings.warn(
fatal: not a git repository (or any parent up to mount point /workspace)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2022-05-07 16:42:40,199 - mmdet - INFO - Environment info:
(...snip...)
2022-05-07 16:43:08,063 - mmdet - INFO - Epoch [1][50/1175]     lr: 1.978e-03, eta: 1:53:59, time: 0.487, data_time: 0.047, memory: 3786, loss_rpn_cls: 0.4788, loss_rpn_bbox: 0.1043, loss_cls: 0.9839, acc: 88.6504, loss_bbox: 0.0727, loss: 1.6396
(...snip...)
```

学習が開始できたら動作としては問題ないはずです

### 02. イメージをregistryに登録する

リモートマシンで作成したイメージを利用できるようにします

#### Web UIにログインする

[https://registry.dev.ai-ms.com/](https://registry.dev.ai-ms.com/) にログインして下さい(IPAのユーザー名でログインできます)

※ [Portus](http://port.us.org/) は `docker registry` に認証をつけるサービスですが、今回はIPA(LDAP)認証に対応しています

Webインターフェースからは作成したイメージなどを確認することができます

#### Docker Registryにログインする

一般の `docker registry` と同様に `docker login` での認証が必要となります

```
$ docker login registry.dev.ai-ms.com
Username: [IPAのユーザー名]
Password: [IPAのパスワード]
Authenticating with existing credentials...
WARNING! Your password will be stored unencrypted in /home/ykato/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

※ `docker logout registry.dev.ai-ms.com` でログアウトすることができます

#### Dockerイメージをアップロードする

作成したイメージの `IMAGE ID` を確認し、`registry.dev.ai-ms.com` 上でのイメージ名を付与します

```
$ docker images
REPOSITORY                                                TAG         IMAGE ID       CREATED          SIZE
clearml-example-mmdetection_clearml_example_mmdetection   latest      0254cfe68a0a   20 minutes ago   14.8GB
(...snip...)
```

`0254cfe68a0a` が作成したイメージのIDです

```
$ docker tag 0254cfe68a0a registry.dev.ai-ms.com/[IPAのユーザー名]/clearml-example-mmdetection:latest
$ docker push registry.dev.ai-ms.com/[IPAのユーザー名]/clearml-example-mmdetection
Using default tag: latest
The push refers to repository [registry.dev.ai-ms.com/[IPAのユーザー名]/clearml-example-mmdetection]
02a697271783: Preparing
(...snip...)
latest: digest: sha256:839bdc18c4c8b3a881de0df9ad6276b10e2ea162d43b1e77fa656c357a40499a size: 10437
```

これでDockerイメージをリモートから参照することができるようになりました

## 4. `clearml-task` 用のコードを用意する

`clearml-task` での実行では開始スクリプトはpythonコードである必要があります。
適切にログを取得しつつ、複数のGPUを使うために分散させようとすると一筋縄ではいきません。

mm系のツールセットを含めて、pytorchのトレーニングでは、一般に `train.py` のような単独GPUで動作するコードを `torch.distributed.launch` や 今では `torchrun` といったpytorchのモジュールを使って分散処理の制御を行います。いづれにしても `shell` を利用しているのですが、これをpythonコードで実現しなければなりません。

そこで [pytorch/torch/distributed/run.py](https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py) から `torchrun` の実行コード自体を改変して作り込みます。

### 01. トレーニング開始コードを作成する

`mmdetection/clearml_train.py` を参照して下さい。

基本的には `tools/dist_train.sh` と同様の機能をするコードをpythonで実現しているだけです。

#### ポイント1: Non positional arguments　を無くす

改変しなければならない最大ポイントは `clearml-task` が無名の引数を渡せないことに起因します。

`training_script`, `training_script_args` の引数を削除します

以下のようにコメントアウトします

```
    #
    # Positional arguments.
    #

    # parser.add_argument(
    #     "training_script",
    #     type=str,
    #     help="Full path to the (single GPU) training program/script to be launched in parallel, "
    #     "followed by all the arguments for the training script.",
    # )

    # Rest from the training program.
    # parser.add_argument("training_script_args", nargs=REMAINDER)
```

#### ポイント2: データセットをトレーニング前に設置する

これは実行前にシェルスクリプトを使うことでも実現できますが、パラメータ化した方が管理できて良いです。

`dataset_id`, `dataset_path` の引数を取るようにして、ClaerMLの `Dataset` モジュールを使って、`dataset_path` に配置します

以下のように引数を追加します

```
    parser.add_argument(
        '--dataset_id',
        type=str,
        default="",
    )
    parser.add_argument(
        '--dataset_path'
        type=str,
        default="",
    )
```

`run()` より前にデータセットをダウンロードします

```
    Dataset.get(dataset_id=args.dataset_id).get_mutable_local_copy(args.dataset_path)
```

#### ポイント3: 設定ファイルを指定する

設定はファイルにまとめて `clearml-task` では、その設定ファイルを指定するようにすればトレーニングの管理をコードで実現できます。

設定ファイルを引数に追加します

```
    parser.add_argument(
        '--config',
        type=str,
        default="",
    )
```

`args` に対して、`training_script` と `training_script_args` を設定します。

```
    os.environ['PYTHONPATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    args.training_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    args.training_script_args = ['--seed', '0', '--launcher', 'pytorch', args.config]
```

`PYTHONPATH` を設定して`mmdetection`のフォルダーを追加しないと`mmdet`などのimoprtが通らないので注意して下さい

※ どうしても可変引数を渡したいのであれば、ここで上手く受け渡しをしましょう

### 02. コミットしておく

clearmlでの差分読み取りは*登録されているコードに対する変更のみ*です。
新規に作成したファイルは`git add/commit`をしておかないと、実行時に参照できないので注意して下さい。

```
$ git add tools/clearml_train.py
$ git commit -m 'add training script'
$ git push
```

### 03. 手元のマシンで実行を試す

最初の実行で上手くいくことは少ないので、`clearml-agent` を手元のPCで実行して動作を確認しましょう。

[ClearML Agent](https://clear.ml/docs/latest/docs/clearml_agent) を参考にセットアップして下さい。

#### 設定ファイル

`~/clearml.conf` は通常の`clearml`と設定を共有できます。
s3のcredentialと以下のagent用の設定を追加すれば十分です。

```
agent {
    git_user: "aim-rnd"
    git_pass: "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"

    extra_docker_arguments:  ["--shm-size=8gb"]
}
```

#### queueを作成する

[GUI](http://app.clearml.dev.ai-ms.com/workers-and-queues/queues) より自分用のqueueを作成します。

#### Agentの開始

```
$ clearml-agent daemon --gpus all --queue [queue名] --docker
(...snip...)
Worker "clearml-client01:gpuall" - Listening to queues:
+----------------------------------+-----------+-------+
| id                               | name      | tags  |
+----------------------------------+-----------+-------+
| 9e925d83222e41b2b4bad8857f86559e | [queue名] |       |
+----------------------------------+-----------+-------+

Running in Docker mode (v19.03 and above) - using default docker image: nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

Running CLEARML-AGENT daemon in background mode, writing stdout/stderr to /tmp/.clearml_agent_daemon_outigpvkain.txt
```

停止する場合は `Ctrl+C` として下さい

#### taskの実行

ログを確認するには `tail -f` でagentの出力をモニターします。

```
$ tail -f /tmp/.clearml_agent_daemon_outigpvkain.txt
(...snip...)
```

タスクは以下のコマンドで実行できます。

コード外から実行できるようにして、

※1 データセットはVOCデータセットを1/10にしたサブセットを作成して動作テスト用に用意したものです
※2 `--packages` を省略するには`mmdetection`内のコードを修正する必要があるので後述します
※3 `dataset_path` がディレクトリ外に配置されるようですが、実際には実行ディレクトリに対して記述するので `datasets.voc0712.py` の `data_root` に合わせる必要があります

```
$ cd [GIT-ROOT]
$ clearml-task --project [適当なプロジェクト名] \
               --name [適当な実験名] \
               --script mmdetection/tools/clearml_train.py \
               --args dataset_id=25d3c555841a4e5583861bc74d44b0de dataset_path=data/VOCdevkit config=mmdetection/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py \
               --branch main \
               --queue [queue名] \
               --docker registry.dev.ai-ms.com/[IPAユーザー名]/clearml-example-mmdetection \
               --packages boto3
```

実行時の環境設定が想像と違ったり、`clearml-task` の使い方も含めて最初は難しいかも知れませんが根気よく試して動作するようにします

### 04. 本番環境で実行する

本番は手元のマシンで試したものと変わらないですが、GPUの数に合わせて `--args` に設定を加えて下さい

`standalone=True nproc_per_node=[GPU数]`

※ `torchrun`の引数と同じです

## 5. トレーニングをログする

### 1. Config

`task.upload_artifact` を使うことで任意のオブジェクトをClearML上に保管することができます。

`mmcv.Config` はそのままだと見づらい形式での出力なので`pretty_text`メソッドで読み易くしてテキストで保持しています。

```
# tools/train.py:227

    # save training config
    from clearml import Task
    task = Task.current_task()
    task.upload_artifact(name='Config', artifact_object=cfg.pretty_text)
```

### 2. TensorboardLoggerHook (lossなど)

`custom_hooks` に `TensorboardLoggerHook` を設定することで、Tensorboardに向けたログが出力されます。
TensorboardのログはClearMLは自動で収集し `RESULTS>SCALARS` に描画されます。

```
# mmdetection/configs/_base_/default_runtime.py: 7
    dict(type='TensorboardLoggerHook'), # 有効化する
```

### 3. ベストモデル

ベストモデルを恒久的に残すようにするには、細かい設置が必要です。

`evaluation` に `save_best='auto'` を追加すると、指定条件でbestモデルを残すようになります。

```
# mmdetection/configs/_base_/datasets/voc0712.py:55
evaluation = dict(save_best='auto', interval=1, metric='mAP')
```

`mmdetection/mmdet/core/evaluation/eval_hooks.py` を編集して、best時のモデルをアップロードするようにします。

```
# mmdetection/mmdet/core/evaluation/eval_hooks.py: 11
from clearml import Task, OutputModel

# mmdetection/mmdet/core/evaluation/eval_hooks.py: 33, 77
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

        self.task = Task.current_task()
        self.output_model = OutputModel(task=self.task, framework='pytorch')

# mmdetection/mmdet/core/evaluation/eval_hooks.py: 62, 133
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
                if self.best_ckpt_path and self.file_client.isfile(self.best_ckpt_path):
                    step = runner.epoch if self.by_epoch else runner.iter
                    self.output_model.update_weights(weights_filename=self.best_ckpt_path, target_filename=f'best_{self.key_indicator}.pth', iteration=step)
```

## テクニック集

### 1. `--requirements or --packages` を省略する

`clearml-task` のバグ的挙動で「`--requirements or --packages` はいづれかが指定されていない場合、GITROOTにある `requirements.txt` を自動的に適用しようとする」というものがあります。それだけなら空のファイルを用意すれば問題ないのですが「実際に使用される `requirements.txt` は実行スクリプトから近いものを探す」という動作をし、`mmdetection/requirements.txt` を実行しようとします。そこで更に`mmdetection`フォルダー内ではなく`/tmp`以下に`requirements.txt`がコピーされ実行されるので失敗するという事態が生じます。

とてもややこしい動きをするので諦めて `--packages boto3` のような無害な引数を作るのが簡単ですが、以下の方法で回避できます。

1. `GITROOT` に 空の `requirements.txt` を配置する
2. `mmdetection/requirements.txt` の内容をコメントアウトする

この２つを設定することで `clearml-task` 実行時の引数を減らすことができます
