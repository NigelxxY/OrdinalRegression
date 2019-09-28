## 参数说明

#### 数据预处理

运行数据预处理的方法在*prepocess.sh*文件中，   
运行的方法为：
```bash
nohup python preprocess.py start_datetime > log_file &
```

其中start_datetime格式为**YYMMDD**，表示数据预处理的最开始一个日期，然后直接读取一个月的数据进行处理。  
其余参数的表示均以方法的变量的形式存放在*preprocess.py*文件中， 具体文件如下： 
- root_path：表示grib数据的根目录文件（其目录下是年）；
- micaps_path：表示实况micaps数据的根目录文件；
- features_save_path：特征保存的路径；
- labels_save_path：实况数据生成的标签的保存路径；
- normalKeys：有多个高度的特征名组成的一个列表，其每个对应`levelList`的每个高度的特征都会被读取；
- singleLevelKeys：只有一个高度的特征名组成的列表；
- precipitationKeys：降水类的特征名组成的列表；
- levelsList：不同的高度组成的列表，会针对`normalKeys`中的每个特征在不同高度上进行数据的读取；
- locBound：表示经纬度的上下限范围的元祖，四个变量分别表示纬度下限、纬度上限、经度下限和经度上限。  

可以改变这些参数来适应定制的需求。

#### 模型训练

训练模型可以直接运行：
```bash
python trainer.py --out='out_file' \
                               --featurePath='feature_save_path' \
                               --labelPath='label_save_path' \
                               --alpha=alpha \
                               --beta=beta \
                               --lr=lr \
                               --epochs=max_epochs \
                               --weight_decay=weight_decay \
                               --gpuid='gpu_id' \
                               --interval=interval \
                               --batch_size=batch_size \
                               --featureNums=featureNums \
                               --noiseStd=noiseStd \
                               --spaces=spaces
```   

针对这些参数的说明如下：

- `--out`：log输出的文件路径根目录，所有的log文件都会根据运行时的时间戳生成一个文件，每次实验的时候的日志文件都在根据这个时间戳生成的文件内；
- `--featurePath`：对应的数据预处理中的`features_save_path`；
- `--labelPath`：对应的数据预处理中的`labels_save_path`；
- `--alpha`：损失函数中的一个超参数，用于调节交叉熵损失函数；
- `--beta`：损失函数中的另一个超参数，用于调节EMD损失函数；
- `--lr`：优化中的学习率；
- `--weight_decay`：权重的惩罚系数，每n次优化中对权重进行一次惩罚；
- `--gpuid`：制定使用哪几个GPU，若只使用CPU则设定为'-1'；
- `--interval`：多少次训练进行一次测试进行验证；
- `--batch_size`：batchsize的大小，根据内存大小来进行设定；
- `--featureNums`：训练过程中用到了多少特征；
- `--noiseStd`：训练中添加随机噪声所服从的高斯分布的标准差；
- `--spaces`：数轴上的分割间隔大小。

这些参数都可以在*train.py*文件中的`if __name__ == "__main__":`中进行修改`parser.add_argument("--some_parameters", default="")`某个参数中的`default`的值来达到声明的效果。