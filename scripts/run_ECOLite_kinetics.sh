



##################################################################### 
# Train on local machine
if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
    cd $PBS_O_WORKDIR
fi


##################################################################### 
# Parameters!
mainFolder="net_runs"
subFolder="ECO_lite_run1"
snap_pref="eco_lite"

train_path="list/kinetics_train.txt"
val_path="list/kinetics_val.txt"

#############################################
#--- training hyperparams ---
dataset_name="kinetics"
netType="ECO"
batch_size=15
learning_rate=0.001
num_segments=16
dropout=0.3
iter_size=4
num_workers=5

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

##################################################################### 
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 15 30 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts finetune --no_partialbn --nesterov "True" --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt    

else
     echo "Training with initialization"

python3 -u main.py ${dataset_name} RGB ${train_path} ${val_path} --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --lr_steps 15 30 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type identity --eval-freq 1 --rgb_prefix img_ --pretrained_parts finetune --no_partialbn --nesterov "True" 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

fi

##################################################################### 


