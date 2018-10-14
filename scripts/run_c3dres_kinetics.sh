

##################################################################### 
# Train on local machine
if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
    cd $PBS_O_WORKDIR
fi


##################################################################### 
# output folder setting!
mainFolder="net_runs"
subFolder="C3Dresnet18_run1"
snap_pref="C3DResNet18_16F"


### data list path #####
train_path="/list/kinetics_train.txt"
val_path="/list/kinetics_val.txt"

#############################################
#--- training hyperparams ---
dataset_name="kinetics"
netType="C3DRes18"
batch_size=32
learning_rate=0.001
num_segments=16
dropout=0.3
iter_size=4
num_workers=4


##################################################################### 
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training

echo "Current network folder: "
echo ${mainFolder}/${subFolder}


##################################################################### 
# Find the latest checkpoint of network 
checkpointIter="$(ls ${mainFolder}/${subFolder}/*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
##################################################################### 


echo "${checkpointIter}"


### Start training - continue training or start from begining ###
##################################################################### 
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

python3 main.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 30 60 --epochs 80 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts 3D --no_partialbn --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt    

else
     echo "Training with initialization"

python3 main.py ${dataset_name} RGB ${train_path} ${val_path} --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 30 60 --epochs 80 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts 3D --no_partialbn 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

fi

##################################################################### 


