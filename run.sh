export SLURM_LOCALID=0
export SLURM_NODEID=0
GPU_ID=0
init_method="tcp://127.0.0.1:3456"${GPU_ID}


image_encoder_list=("resnet50" "densenet121" "densenet161" "densenet169" "densenet201" "resnet50" "resnet101" "resnet152" "vit_base_patch32_224" "vit_large_patch32_224_in21k" "vit_base_patch32_224_clip_laion2b")


# dataset: GSE240429
max_epochs=25
batch_size=256
lr=0.001
aggregation_method=average
top_K=100
path_dir=None
pt_file=None
dataset=GSE240429  # "GSE240429", "her2st"
model_name=ours
for Recurrence in 1 2 3; do
#  for image_encoder in ${image_encoder_list[@]}; do
  for image_encoder in densenet121; do
    path_dir=${model_name}-${dataset}-${image_encoder}-${Recurrence}
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --path_dir ${path_dir} --init_method ${init_method} --dataset ${dataset} --image_encoder ${image_encoder} --max_epochs ${max_epochs} --batch_size ${batch_size} --lr ${lr}
    for pt_file in ./result/${path_dir}/*.pt; do
      CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
    done
  done
done


# dataset: her2st
max_epochs=25
batch_size=1
lr=0.0002
aggregation_method=average
top_K=100
path_dir=None
pt_file=None
dataset=her2st  # "GSE240429", "her2st"
model_name=ours
#for image_encoder in ${image_encoder_list[@]}; do
for image_encoder in densenet121; do
  path_dir=${model_name}-${dataset}-${image_encoder}
  CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py --path_dir ${path_dir} --init_method ${init_method} --dataset ${dataset} --image_encoder ${image_encoder} --max_epochs ${max_epochs} --batch_size ${batch_size} --lr ${lr}
  for pt_file in ./result/${path_dir}/*.pt; do
    for fold in 0 6 12 18 24 27 31 33; do
      CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --path_dir ${path_dir}  --pt_path ${pt_file} --dataset ${dataset} --fold ${fold} --image_encoder ${image_encoder} --aggregation_method ${aggregation_method} --top_K ${top_K}
    done
  done
done


# HisToGene her2st
cd ./method/HisToGene/
ckpt_name=../../checkpoint/HisToGene/her2st_HisToGene_best
ckpt=${ckpt_name}.ckpt
log=${ckpt_name}.txt
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_HisToGene.py --train train --ckpt ${ckpt} >> ${log}
cd ../../
#exit 0

# ST-Net her2st
cd ./method/ST-Net/
ckpt_name=../../checkpoint/ST-Net/her2st_ST-Net_best
ckpt=${ckpt_name}.ckpt
log=${ckpt_name}.txt
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./main_ST-Net.py --train train --ckpt ${ckpt} >> ${log}
cd ../../
exit 0


