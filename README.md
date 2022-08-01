
# DynamicFL
The implementation of DynamicFL in the paper "Dynamic-Fusion-Based Federated Learning for COVID-19 Detection"

## Getting Started
### Clone the repo
```shell
git clone https://github.com/upcbdipt/DynamicFL.git && cd DynamicFL
```

### Installing dependencies
Using the following command to install dependencies (recommend using a virtualenv to manage your environment).
```shell
pip install -r requirements.txt
```

### Training
#### Downloading data
The origin dataset can be downloaded from [CT](https://github.com/UCSD-AI4H/COVID-CT), [Kaggle-Xray](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database), [Github-Xray](https://github.com/agchung/Figure1-COVID-chestxray-datase).

#### Extracting origin.zip
Extract zip file to data/origin. The structure of the data folder should be
```
data/
  origin/
    train/
      CT/
        COVID-19/*.png
        NORMAL/*.png
      Xray/
        COVID-19/*.png
        NORMAL/*.png
        Viral Pneumonia/*.png
    test/
      ...
``` 


#### Preprocessing data
```shell
python tools/preprocess.py -dn 1 -src data/origin -dst data/preprocess/ -m False
```
The optional parameters ```-dn 1``` refers to the difference dataset idx in this paper. The ```-src data/origin``` refers to the origin dataset dir. The ```-dst data/preprocess/``` refers to the dir saving the preprocessed data. The ```-m False``` represents whether merge the same classes in the CT and Xray images data. If you merge the same classes, you should change the ```num_classes``` of model.

#### Starting server
```
python server.py -b resnet50 -cs 50 -alg DynamicFL -ip 127.0.0.1 -port 9770
```
We offer three different models for training, i.e. ```resnet50```, ```resnet101```, ```ghost_net```. The parameter ```-cs``` refers the communication epochs for aggregation. The ```-alg FedAvg/DynamicFL``` presents the aggregation algorithm. You should change the ip and port by the ```-ip``` and ```-port``` optional parameters for your network environment.

#### Starting clients
```
python client.py -alg DynamicFL -c 1 -d data/preprocess/N1 -ip 127.0.0.1 -p 9081 -su http://127.0.0.1:9770
```
If you want to start clients, you should change the parameters ```-c/--client```, ``` -d/--datadir ```, ``` -p/--port ```, ``` -ip ```, ``` -su/--server_url ```.


## Citation
If you use this work, please cite:
```
@article{zhang2021dynamic,
  title={Dynamic-fusion-based federated learning for COVID-19 detection},
  author={Zhang, Weishan and Zhou, Tao and Lu, Qinghua and Wang, Xiao and Zhu, Chunsheng and Sun, Haoyun and Wang, Zhipeng and Lo, Sin Kit and Wang, Fei-Yue},
  journal={IEEE Internet of Things Journal},
  volume={8},
  number={21},
  pages={15884--15891},
  year={2021},
  publisher={IEEE}
}
```

## License
DynamicFL is distributed under [Apache 2.0 licensed](http://www.apache.orgenses/LICENSE-2.0).

Contact: Weishan Zhang (zhangws@upc.edu.cn)
