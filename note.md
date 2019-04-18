提取特征的repos在二楼服务器：/bak/Git_Repos_Summer/ExtractFeature/
### 需要更改的地方：
- 把要提取特征的图像文件夹路径改一下
- 把要保存的feature.txt名称改一下
**Bug1**:
  File "testcode_ExtractFeatures.py", line 73, in <module>
    alexnet_model = alexnet()
  File "testcode_ExtractFeatures.py", line 70, in alexnet
    model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
  File "/usr/local/lib/python3.5/dist-packages/torch/nn/modules/module.py", line 723, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for AlexNet:
        Missing key(s) in state_dict:  ......
[DEBUG1参考
](https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686)

解决方法：在王发我的原pytorch网站上 
def alexnet（pretrained=false） 我改成了true。。。
如果网络里面的层不能改，就需要自己另加一个def  

https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/15

按照AlexNet重写了一个网络，继承了AlexNet原来的层
```
model_urls = { 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
model = torchvision.models.alexnet()
model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
class MyAlexNetFeatureExtractor(nn.Module):
    def __init__(self, alexnet, transform_input=False):
        super(MyAlexNetFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            alexnet.features[0],
            alexnet.features[1],
            alexnet.features[2],
            alexnet.features[3],
            alexnet.features[4],
            alexnet.features[5],
            alexnet.features[6],
                                     )
    def forward(self, x):
        x = self.features(x)
        return x
AlexNet_Feature =MyAlexNetFeatureExtractor(model)
```
RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.

###### detach
[AssertionError: nn criterions don’t compute the gradient w.r.t. targets - please mark these variables as volatile or not requiring](
https://discuss.pytorch.org/t/assertionerror-nn-criterions-dont-compute-the-gradient-w-r-t-targets-please-mark-these-variables-as-volatile-or-not-requiring-gradients/21542
)
#### 要用SegNet测试CycleGAN生成的图像，又遇到需要caffe的问题，[按照这个帖子安装了caffe]( https://blog.csdn.net/yhaolpz/article/details/71375762)又get一个编译技能：
 
```
make clean
cd caffe
mkdir build
cmake ..
make all -j8
```
