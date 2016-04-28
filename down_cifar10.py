import os
if os.path.exists('1_torch_intro/cifar10-test.t7') or os.path.exists('cifar10-test.t7'):
    print "cifar10 already exists"
else:
    os.system("/usr/bin/wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip -O 1_torch_intro/cifar10torchsmall.zip")
    os.system("/usr/bin/unzip 1_torch_intro/cifar10torchsmall.zip -d 1_torch_intro")
    print "Successfully retrieved"