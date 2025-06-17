# 扩散模型项目

## DDPM原理

参考文档：https://blog.csdn.net/weixin_42392454/article/details/137458318

扩散模型中最重要的思想根基是马尔可夫链，它的一个关键性质是平稳性。即如果一个概率随时间变化，那么在马尔可夫链的作用下，它会趋向于某种平稳分布，时间越长，分布越平稳。

### 数学理论

### 1.马尔可夫原理

**任意联合概率分布** p(x_1, x_2, ..., x_T)$都可以写成：
$$
p(x_1, x_2, ..., x_T) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2) \cdots p(x_T \mid x_1, ..., x_{T-1})
$$
马尔可夫过程定义，其只与前一个时刻有关，故上述公式改为如下：
<img src="./assets/1.png" alt="image-20250616161850837" style="zoom: 50%;" />       

### 2.重参数化过程

这里补充说明一下，重参数的过程，假设从某个正态分布N (μ,σ2 ∗ I )采样一个X的话，它可以等价于，从一个标准正态分布N(0,I)去采样一个Z，然后利用Z去生成X：

<img src="./assets/2.png" alt="image-20250616210629947" style="zoom:50%;" />

### 原理理解（非数学）

Stable Diffusion 分为 Diffusion 和 Reverse 两个阶段。其中 Diffusion 阶段通过不断地对真实图片添加噪声，最终得到一张噪声图片。而 Reverse 阶段，模型需要学习预测出一张噪声图片中的噪声部分，然后减掉该噪声部分，即：去噪。随机采样一张完全噪声图片，通过不断地去噪，最终得到一张符合现实世界图片分布的真实图片。

![image-20250616155744063](./assets/3.png)

## 前向过程

这个阶段就是不断地给真实图片加噪声，经过T TT步加噪之后，噪声强度不断变大，得到一张完全为噪声地图像。整个扩散过程可以近似看成一次加噪即变为噪声图。
![image-20250618012703631](./assets/5.png)

## 逆向过程

论文中的结论可以知道，这么做的效果比较差，图片是很模糊的，不符合逆扩散的过程，最好还是一步一步推。

![image-20250616202221269](./assets/4.png)

流程：

1.由上图可知，需要由X<sub>t</sub>推理得到X<sub>t-1</sub>,相当于已知X<sub>t</sub>概率，去求X<sub>t-1</sub>的条件概率

* <img src="./assets/6.png" alt="image-20250616205323356" style="zoom: 50%;" />
* 因为我们当前已知的是已知X<sub>t-1</sub>概率，求X<sub>t</sub>的条件概率，然后我们把

## DDIM原理

## 项目复现

## 1.数据预处理$ pbcopy < ~/.ssh/id_rsa.pub

从github下载data读取的python代码：https://github.com/fyu/lsun

```
pip install opencv-python
pip install lmdb
```

然后在anaconda下运行代码：
以下是相对路径形式：

```python
python data.py export ./data --out_dir ./lsun/cat --flat
```

注意这里的路径./data表示data文件夹下面包含data.mdb和lock.mdb两个文件
以下是绝对路径形式：

```python
python data.py export F:/大三春季学期/上海交大/church_val --out_dir F:/大三春季学期/上海交大/data/church_val --flat
```

表示F盘有文件夹LSUN，LSUN文件夹下面有cat文件夹，这里是存储的文件；要输出到F盘下面的data文件夹下面的cat文件夹。此时data.py放在F盘根目录。

## 2.模型

