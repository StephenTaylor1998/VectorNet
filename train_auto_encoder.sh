
python train_auto_encoder.py --data_format cifar10 -a get_vgg_encoder -b 512 -j 32 -c 10 --epoch 400 --dist-url 'tcp://127.0.0.1:8889' \
--dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset/ --weight-decay 5e-4 \
#--resume ./data/model_best.pth.tar
