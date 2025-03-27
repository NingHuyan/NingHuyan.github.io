# Bio of Ning Huyan (呼延宁)                               
<p align="center">
 <img src="huyan.jpg" width="50%" height="50%" align='center'/>
</p>

  从2024年9月到现在，我就职于清华大学自动化系（博士后），从事动作生成、高光谱影像异常检测相关研究工作。当前我的合作导师是[**刘烨斌**](http://www.liuyebin.com/)教授。我与西安电子科技大学**权豆**副教授保持良好合作关系，并共同开展相关研究工作。2022年5月到2024年8月我曾就职于西安商汤科技有限公司，担任研究员，期间有过一段愉快的工作经历，主要从事基于光学刚体的动作捕捉相关研发相关工作。2019年11月到2020年11月曾在法国格勒诺布尔法国国家信息与自动化研究所访问交流，与[**jocelyn chanussot**](https://jocelyn-chanussot.net/)愉快合作。2011年到2022年在西安电子科技大学分别取得学士和博士学位。攻读博士期间主要跟随西安电子科技大学人工智能学院[**张向荣**](https://web.xidian.edu.cn/xrzhang/)教授, 从事高光谱影像异常检测相关研究工作。


## 研究兴趣:

（高光谱影像）异常检测、稀疏刚体动作捕捉，动作生成。主要涉及技术为深度学习模型（Transformer，ResNet）以及字典学习模型（Low Rank， Dictionary Learning）。

## 工作经历:

* 09/2024 - 至今:    博士后， 清华大学自动化系，北京，中国。

* 08/2024 - 05/2022: 算法研究员， 西安商汤智能科技有限公司， 陕西西安，中国。

* 06/2022 - 09/2011: 学士—博士， 西安电子科技大学，陕西西安，中国。

## 论文发表:
* **Ning Huyan**, Dou Quan, Xiangrong Zhang, Xuefeng Liang, Jocelyn Chanussot, Licheng Jiao. **Unsupervised outlier detection using memory and contrastive learning**. EEE Transactions on Image Processing 31, 6440-6454. 2022/10/10.
  
<p align="center">
 <img src="MCOD.png" width="50%" height="50%"/>
</p>

+ **Abstract:** Outlier detection is to separate anomalous data from inliers in the dataset. Recently, the most deep learning methods of outlier detection leverage an auxiliary reconstruction task by assuming that outliers are more difficult to recover than normal samples (inliers). However, it is not always true in deep auto-encoder (AE) based models. The auto-encoder based detectors may recover certain outliers even if outliers are not in the training data, because they do not constrain the feature learning. Instead, we think outlier detection can be done in the feature space by measuring the distance between outliers’ features and the consistency feature of inliers. To achieve this, we propose an unsupervised outlier detection method using a memory module and a contrastive learning module (MCOD). The memory module constrains the consistency of features, which merely represent the normal data. The contrastive learning module learns more discriminative features, which boosts the distinction between outliers and inliers. Extensive experiments on four benchmark datasets show that our proposed MCOD performs well and outperforms eleven state-of-the-art methods.  [**PDF**](https://ieeexplore.ieee.org/abstract/document/9913887) [**CODE**](https://github.com/huyanning/MCOD)
  
* **Ning Huyan**, Xiangrong Zhang, Dou Quan, Jocelyn Chanussot, Licheng Jiao. **AUD-Net: A unified deep detector for multiple hyperspectral image anomaly detection via relation and few-shot learning**. IEEE Transactions on Neural Networks and Learning Systems 35 (5), 6835-6849. 2022/10/27.
  
<p align="center">
 <img src="AUD.png" width="50%" height="50%" align='center'/>
 </p>
 
+ **Abstract:** This article addresses the problem of the building an out-of-the-box deep detector, motivated by the need to perform anomaly detection across multiple hyperspectral images (HSIs) without repeated training. To solve this challenging task, we propose a unified detector [anomaly detection network (AUD-Net)] inspired by few-shot learning. The crucial issues solved by AUD-Net include: how to improve the generalization of the model on various HSIs that contain different categories of land cover; and how to unify the different spectral sizes between HSIs. To achieve this, we first build a series of subtasks to classify the relations between the center and its surroundings in the dual window. Through relation learning, AUD-Net can be more easily generalized to unseen HSIs, as the relations of the pixel pairs are shared among different HSIs. Secondly, to handle different HSIs with various spectral sizes, we propose a pooling layer based on the vector of local aggregated descriptors, which maps the variable-sized features to the same space and acquires the fixed-sized relation embeddings. To determine whether the center of the dual window is an anomaly, we build a memory model by the transformer, which integrates the contextual relation embeddings in the dual window and estimates the relation embeddings of the center. By computing the feature difference between the estimated relation embeddings of the centers and the corresponding real ones, the centers with large differences will be detected as anomalies, as they are more difficult to be estimated by the corresponding surroundings. Extensive experiments on both the simulation dataset and 13 real HSIs demonstrate that this proposed AUD-Net has strong generalization for various HSIs and achieves significant advantages over the specific-trained detectors for each HSI.  [**PDF**](https://ieeexplore.ieee.org/abstract/document/9931456) [**CODE**](https://github.com/huyanning/AUD-Net)
  
* **Ning Huyan**, Xiangrong Zhang, Huiyu Zhou, Licheng Jiao. **Hyperspectral anomaly detection via background and potential anomaly dictionaries construction**. IEEE Transactions on Geoscience and Remote Sensing 57 (4), 2263-2276. 2018/11/1.
  
<p align="center">
 <img src="ABD.png" width="50%" height="50%" align='center'/>
  </p>
  
+ **Abstract:** In this paper, we propose a new anomaly detection method for hyperspectral images based on two well-designed dictionaries: background dictionary and potential anomaly dictionary. In order to effectively detect an anomaly and eliminate the influence of noise, the original image is decomposed into three components: background, anomalies, and noise. In this way, the anomaly detection task is regarded as a problem of matrix decomposition. Considering the homogeneity of background and the sparsity of anomalies, the low-rank and sparse constraints are imposed in our model. Then, the background and potential anomaly dictionaries are constructed using the background and anomaly priors. For the background dictionary, a joint sparse representation (JSR)-based dictionary selection strategy is proposed, assuming that the frequently used atoms in the overcomplete dictionary tend to be the background. In order to make full use of the prior information of anomalies hidden in the scene, the potential anomaly dictionary is constructed. We define a criterion, i.e., the anomalous level of a pixel, by using the residual calculated in the JSR model within its local region. Then, it is combined with a weighted term to alleviate the influence of noise and background. Experiments show that our proposed anomaly detection method based on potential anomaly and background dictionaries construction can achieve superior results compared with other state-of-the-art methods.  [**PDF**](https://ieeexplore.ieee.org/abstract/document/8519775)
  
* **Ning Huyan**, Xiangrong Zhang, Dou Quan, Jocelyn Chanussot, Licheng Jiao. **Cluster-memory augmented deep autoencoder via optimal transportation for hyperspectral anomaly detection**. IEEE Transactions on Geoscience and Remote Sensing 60, 1-16. 2022/6/6.
  
<p align="center">
 <img src="OTCMA.png" width="50%" height="50%" align='center'/>
  </p>
  
+ **Abstract:** Hyperspectral anomaly detection (AD) aims to detect objects significantly different from their surrounding background. Recently, many detectors based on autoencoder (AE) exhibited promising performances in hyperspectral AD tasks. However, the fundamental hypothesis of the AE-based detector that anomaly is more challenging to be reconstructed than background may not always be true in practice. We demonstrate that an AE could well reconstruct anomalies even without anomalies for training, because AE models mainly focus on the quality of sample reconstruction and do not care if the encoded features solely represent the background rather than anomalies. If more information is preserved than needed to reconstruct the background, the anomalies will be well reconstructed. This article proposes a cluster-memory augmented deep autoencoder via optimal transportation for hyperspectral anomaly detection (OTCMA) clustering for hyperspectral AD to solve this problem. The deep clustering method based on optimal transportation (OT) is proposed to enhance the features consistency of samples within the same categories and features discrimination of samples in different categories. The memory module stores the background’s consistent features, which are the cluster centers for each category background. We retrieve more consistent features from the memory module instead of reconstructing a sample utilizing its own encoded features. The network focuses more on consistent feature reconstruction by training AE with a memory module. This effectively restricts the reconstruction ability of AE and prevents reconstructing anomalies. Extensive experiments on the benchmark datasets demonstrate that our proposed OTCMA achieves state-of-the-art results. Besides, this article presents further discussions about the effectiveness of our proposed memory module and different criteria for better AD.  [**PDF**](https://ieeexplore.ieee.org/abstract/document/9789180) [**CODE**](https://github.com/huyanning/OTCMA-Net)




## 奖项:


## 联系方式: 

**Email:** n-hy@mail.tsinghua.edu.cn， ninghuyan1121@gmail.com

**Address:** Xi'an, Shannxi, China.

<img src="exp5_rgb_orbit-mild_.gif" width="50%" height="50%" align='left' style='width:270px;height:480px'/>
<img src="optimusprime8-rgb-short.gif" width="50%" height="50%" align='middle' style='width:270px;height:480px'/>
<img src="starscream4_rgb.gif" width="50%" height="50%" align='right' style='width:270px;height:480px'/>

<img src="OptimusPrime_8_opt.gif" width="100%" height="100%" align='middle' style='width:800px;height:800px'/>
<img src="StarScream_4_opt_2.gif" width="100%" height="100%" align='middle' style='width:800px;height:800px'/>

## News and Activities:

**Mar 2025:** One paper titled with **Cross-Rejective Open-Set SAR Image Registration** is accepted by ***CVPR 2025***.


