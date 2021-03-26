#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/imagenet2012/

#python train_imagenet.py -d imagefolder -a resnet18 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 /home/aistudio/Desktop/datasets/ILSVRC2012/

#python train_imagenet.py -d cifar10 -a resnet18 -b 64 -j 8 -c 10 --dist-url 'tcp://127.0.0.1:8889' --dist-backend 'nccl' \
#--multiprocessing-distributed --world-size 1 --rank 0 ./data/dataset

#python train_imagenet.py -d cifar10 -a resnet50_tiny -b 32 -j 4 -c 10 ./data/dataset

#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 ./data/dataset
#python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 ./data/dataset \
#--resume ./data/checkpoint.pth.tar --epoch 150

python train_imagenet.py -d fashion_mnist -a resnet50_tiny_c1 -b 128 -j 4 -c 10 -e ./data/dataset \
--resume ./data/checkpoint.pth.tar


