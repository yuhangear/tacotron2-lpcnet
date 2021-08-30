#!/bin/bash

# maowang@ntu, 2020

# this code is designed to do tts using lpcnet feature
# firstly, please extract lpcnet feature
# when do systhesis, please don't use espnet rather than using yourself code.

set -e

. path.sh || exit 1;
#. cmd.sh || exit 1;

# general config
cmd="slurm.pl --quiet"
backend=pytorch
stage=1
stop_stage=1
ngpu=1       # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32 #50        # numebr of parallel jobs
dumpdir=dump # directory to dump full features
verbose=0    # verbose option (if set > 0, get more log)
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# config files
train_config=conf/train_pytorch_tacotron2.v3.yaml # you can select from conf or conf/tuning.
                                               # now we support tacotron2, transformer, and fastspeech
                                               # see more info in the header of each config.
decode_config=conf/decode_pytorch_tacotron2.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best

# exp tag
tag="1a" # tag for managing experiments.

# lpcnet workplace
lpcnet=/media/mipitalk/home/zr511/projects/LPCNet
trans_type="char"

train_set="char_train_no_dev"
dev_set="char_dev"
eval_set="char_eval"
total_set="char_train"

expdir="lpcnet"
data=data

# data preparation firstly you should generate mel spectrogram features
# in data dir:feats.scp(mel feat information)

# generate dump directory
feat_tr_dir=$data/${train_set}/feats
feat_dt_dir=$data/${dev_set}/feats
feat_ev_dir=$data/${eval_set}/feats
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    [ -d $feat_tr_dir ] || mkdir -p $feat_tr_dir
    [ -d $feat_dt_dir ] || mkdir -p $feat_dt_dir
    [ -d $feat_ev_dir ] || mkdir -p $feat_ev_dir
    # make a dev set
    utils/subset_data_dir.sh --last $data/char_train 500 $data/${trans_type}_deveval
    utils/subset_data_dir.sh --last $data/char_deveval 250 $data/${eval_set}
    utils/subset_data_dir.sh --first $data/char_deveval 250 $data/${dev_set}
    n=$(( $(wc -l < data/char_train/utt2spk) - 500 ))
    utils/subset_data_dir.sh --first $data/char_train ${n} $data/${train_set}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in ${train_set} ${eval_set} ${dev_set} ; do
    [ -d $data/$x/s16 ] || mkdir -p $data/$x/s16
    cat $data/$x/wav.scp | while read line; do
      uttid=`echo $line | awk '{print $1}'`
      wav=`echo $line | awk '{print $2}'`
      sox $wav -r 16000 -c 1 -t sw - > $data/$x/s16/${uttid}.s16
    done
    #sed  's/\-t wav \-c /\-t sw \-c /g' $data/$x/wav.scp | cut -d " " -f2 > $data/$x/wavpath
    # for wav in `cat $data/$x/wavpath`; do
    #   sox $wav -r 16000 -c 1 -t sw - > $data/$x/s16/$(basename "$wav" | cut -d. -f1).s16
    # done

    [ -d $data/$x/f32 ] || mkdir -p $data/$x/f32
    for s in $data/$x/s16/*.s16; do
      $lpcnet/dump_data -test $s $data/$x/f32/$(basename "$s" | cut -d. -f1).f32
    done

    [ -d $data/$x/npy ] || mkdir -p $data/$x/npy
    python make_feat.py $data/$x/f32 $data/$x/npy

    [ -d $data/$x/feat ] || mkdir -p $data/$x/feat
    python make_feat_format.py $data/$x/npy $data/$x/feat 
    cat $data/$x/feat/*.scp > $data/$x/feats/feats.scp
  done
fi

# make a dictionary and json data format
dict=$data/lang_1char/${train_set}_units.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # if you don't have a dict please
    # you have to check non-linguistic symbols used in the corpus.
    #echo "stage 1: Dictionary and Json Data Preparation"
    [ -d $data/lang_1char ] || mkdir -p $data/lang_1char
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 --trans_type char $data/char_train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    
    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --trans_type char \
         $data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --trans_type char \
         $data/${dev_set} ${dict} > ${feat_dt_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/feats.scp --trans_type char \
         $data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

# start to train tts am system
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi

# export CUDA_VISIBLE_DEVICES="0,1,2,3"

tts_expdir=$expdir/exp/${expname}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 2: Text-to-speech model training"
    [ -d $tts_expdir ] || mkdir -p $tts_expdir
    tr_json=${feat_tr_dir}/data.json
    dt_json=${feat_dt_dir}/data.json
    ${cmd} --gpu ${ngpu} ${tts_expdir}/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --minibatches ${N} \
           --outdir ${tts_expdir}/results \
           --tensorboard-dir ${tts_expdir}/tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

# decoding tts
if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${tts_expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 3: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${tts_expdir}/results/snapshot.ep.* \
                               --out ${tts_expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in ${eval_set} ${dev_set}; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${feat_dt_dir}/data.json ${outdir}/${dev_set}
	      cp ${feat_ev_dir}/data.json ${outdir}/${eval_set}
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out ${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${tts_expdir}/results/${model} \
                --config ${decode_config}
        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

lpcnetdir=/media/mipitalk/home/zr511/projects/LPCNet
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  for name in ${eval_set}; do
    [ -d ${outdir}/${name}/npy ] || mkdir -p ${outdir}/${name}/npy
    python make_npy.py ${outdir}/${name} ${outdir}/${name}/npy
    [ -d ${outdir}/${name}/f32 ] || mkdir -p ${outdir}/${name}/f32
    python make_f32.py ${outdir}/${name}/npy ${outdir}/${name}/f32

    [ -d ${outdir}/${name}/pcm ] || mkdir -p ${outdir}/${name}/pcm
    [ -d ${outdir}/${name}/wav ] || mkdir -p ${outdir}/${name}/wav
    for f in ${outdir}/${name}/f32/*.f32; do
      uttid=`basename -s .f32 $f`
      $lpcnetdir/lpcnet_demo -synthesis $f ${outdir}/${name}/pcm/${uttid}.pcm
      ffmpeg -f s16le -ar 16k -ac 1 -i ${outdir}/${name}/pcm/${uttid}.pcm ${outdir}/${name}/wav/${uttid}.wav > /dev/null 2>&1
    done

  done
fi
