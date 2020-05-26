### Extract features from ijb-b data set
python2 -W ignore main.py -g -dn ijbb -rm extr_feat -ft -ckp ...model_.._.pth 

### Train the model on VGGFace2 data set with teacher and torchvion-like senet50
python2 -W ignore main.py -g -dn vggface2 -rm train -tm -se -lt -ll 0.1 -lp 0.7 -lr 0.1 -m 0.9 -sv 30 -c -cs 45000 -e 40 -s | tee out.txt

### Train the model on VGGFace2-500
python2 -W ignore main.py -g -dn vggface2-500 -rm train -lr 0.01 -m 0.9 -fn -e 40 -s -sv 50 -sp 10 | tee out.txt