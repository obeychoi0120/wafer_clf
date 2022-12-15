python convnet_baseline.py -ckpt models/convnet_b256_100ep_1013-1755.pt
echo "Baseline Done"

python convnet_baseline.py -augprob 0.25 -ckpt models/convnet_b256_100ep_1013-2034.pt
echo "Aug Done"

python convnet_baseline.py -augprob 0.25 -use-downsample True -ckpt models/convnet_b256_100ep_1014-0004.pt
echo "Downsample Done"

python convnet_baseline.py -augprob 0.25 -use-cweight True -ckpt models/convnet_b256_100ep_1014-0104.pt
echo "Cweight Done"

python convnet_baseline.py -augprob 0.25 -use-downsample True -use-cweight True -ckpt models/convnet_b256_100ep_1014-0351.pt
echo "Downsample, Cweight Done"

python convnet_baseline.py -augprob 0.25 -use-focal True -ckpt models/convnet_b256_100ep_1014-0452.pt
echo "Focal Done"

python convnet_baseline.py -augprob 0.25 -use-focal True -use-downsample True -ckpt models/convnet_b256_100ep_1014-0741.pt
echo "Downsample, Focal Done"

python convnet_baseline.py -augprob 0.25 -use-focal True -use-cweight True -ckpt models/convnet_b256_100ep_1014-0842.pt
echo "Cweight, Focal Done"

python convnet_baseline.py -augprob 0.25 -use-focal True -use-cweight True -use-downsample True -ckpt models/convnet_b256_100ep_1014-1129.pt
echo "Downsample, Cweight, Focal Done"


# python convnet_baseline.py -augprob 0.25
