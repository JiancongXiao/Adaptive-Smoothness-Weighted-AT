# Adaptive Smoothness-weighted Adversarial Training for Multiple Perturbations with Its Stability Analysis

**Jiancong Xiao, Zeyu Qin, Yanbo Fan, Baoyuan Wu, Jue Wang, Zhi-Quan Luo**

**arXiv:** [https://arxiv.org/abs/2210.00557](https://arxiv.org/abs/2210.00557) 

**Workshop version:** [https://openreview.net/pdf?id=qvALKz8BUV](https://openreview.net/pdf?id=qvALKz8BUV)

The Second Workshop on New Frontiers in Adversarial Machine Learning	



## Pretrained Models  
Pretrained models for each of the training methods discussed in the paper are available in the folder `Selected`  
The testing code is automatically designed to pick the models from the folder. 


## Training Code

+ `train.py` - Train the Adversarially Robust Models
  > `gpu_id`  - Id of GPU to be used  - `default = 0`  
  > `model`   - Type of Adversarial Training:  - `default = 9`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9: ADT
  > `batch_size` - Batch Size for Train Set -`default = 128` 

	
**5. Triple Augmentation:**  
`python train.py -model 9`

## Citation
```
@article{xiao2022adaptive,
  title={Adaptive Smoothness-weighted Adversarial Training for Multiple Perturbations with Its Stability Analysis},
  author={Xiao, Jiancong and Qin, Zeyu and Fan, Yanbo and Wu, Baoyuan and Wang, Jue and Luo, Zhi-Quan},
  journal={arXiv preprint arXiv:2210.00557},
  year={2022}
}
```







