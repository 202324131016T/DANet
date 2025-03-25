export SLURM_LOCALID=0
export SLURM_NODEID=0
export LD_LIBRARY_PATH=/mnt/sda/wyl/anaconda/envs/targetDiff/lib:$LD_LIBRARY_PATH


GPU_ID=0
init_method="tcp://127.0.0.1:3456"${GPU_ID}


# ours GSE240429
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0

# ours her2st
max_epochs=25
batch_size=1
lr=0.0002
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/ours-her2st-best.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
exit 0


# *********  dataset: GSE240429  *********

# BLEEP GSE240429
max_epochs=150
batch_size=512
lr=0.001
path_dir=./checkpoint/BLEEP/
pt_file=None
image_encoder=resnet50
top_K=50
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
pt_file=${path_dir}/BLEEP-GSE240429-best.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
#exit 0


# ours GSE240429
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0

# *********  ablation for GSE240429  *********

# YES-1-simple
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=1
aggregation_method=simple
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# YES-50-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=50
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# YES-50-weighted_average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=50
aggregation_method=weighted_average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# YES-100-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# YES-100-weighted_average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=weighted_average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ours-GSE240429-best-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-NO  NO-100-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ablation-NO/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ablation-GSE240429-NO-100-average-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-Densenet  Yes-100-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ablation-Densenet/
pt_file=None
image_encoder=resnet50
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ablation-GSE240429-YES-100-average-Densenet-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-Mamba  Yes-100-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ablation-Mamba/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ablation-GSE240429-YES-100-average-Mamba-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-RKAN  Yes-100-average
max_epochs=25
batch_size=256
lr=0.001
path_dir=./checkpoint/ablation-RKAN/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=GSE240429  # "GSE240429", "her2st"
for Recurrence in 1 2 3; do
pt_file=${path_dir}/ablation-GSE240429-YES-100-average-RKAN-${Recurrence}.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0

# *********  dataset: her2st  *********

# HisToGene her2st
cd ./method/HisToGene/
ckpt_name=../../checkpoint/HisToGene/her2st_HisToGene_best
ckpt=${ckpt_name}.ckpt
log=${ckpt_name}.txt
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_HisToGene.py --train test --ckpt ${ckpt} >> ${log}
cd ../../
#exit 0

# ST-Net her2st
cd ./method/ST-Net/
ckpt_name=../../checkpoint/ST-Net/her2st_ST-Net_best
ckpt=${ckpt_name}.ckpt
log=${ckpt_name}.txt
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_ST-Net.py --train test --ckpt ${ckpt} >> ${log}
cd ../../
#exit 0

# BLEEP her2st
max_epochs=150
batch_size=1
lr=0.001
path_dir=./checkpoint/BLEEP/
pt_file=None
image_encoder=resnet50
top_K=50
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/BLEEP-her2st-best.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ours her2st
max_epochs=25
batch_size=1
lr=0.0002
path_dir=./checkpoint/ours/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/ours-her2st-best.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0

# *********  ablation for her2st  *********

# ablation-Densenet  Yes-100-average
max_epochs=25
batch_size=1
lr=0.0002
path_dir=./checkpoint/ablation-Densenet/
pt_file=None
image_encoder=resnet50
top_K=100
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/ablation-her2st-YES-100-average-Densenet.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-Mamba  Yes-100-average
max_epochs=25
batch_size=1
lr=0.0002
path_dir=./checkpoint/ablation-Mamba/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/ablation-her2st-YES-100-average-Mamba.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0


# ablation-RKAN  Yes-100-average
max_epochs=25
batch_size=1
lr=0.0002
path_dir=./checkpoint/ablation-RKAN/
pt_file=None
image_encoder=densenet121
top_K=100
aggregation_method=average
dataset=her2st  # "GSE240429", "her2st"
for fold in 0 6 12 18 24 27 31 33; do
pt_file=${path_dir}/ablation-her2st-YES-100-average-RKAN.pt
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir} --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
done
#exit 0