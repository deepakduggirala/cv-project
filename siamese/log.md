### 20220422-101221

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=30, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
```

```
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.25, 'squared': True, 'embedding_size': 256, 'lr': 0.0001, 'dense_l2_reg_c': 0.001}
```

- without finetune, l2_c=0.001, train loss is higher than val loss.
- after 20 epochs , val loss flattens out

### 20220422-101827

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=30, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.25, 'squared': True, 'embedding_size': 256, 'lr': 0.0001, 'dense_l2_reg_c': 0.0003}
```

- train loss is higher than val loss, but meets val loss at epoch 30
- val_loss is slightly lower than previous run with c = 0.001


### 20220422-102631

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=30, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.25, 'squared': True, 'embedding_size': 256, 'lr': 0.0001, 'dense_l2_reg_c': 0.0001}
```

- train loss is higher than val loss, but meets val loss at epoch 20
- val_loss is slightly higher than previous run with c = 0.0003




 ### 20220423-155249 

 ```
 Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=40, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 128, 'lr': 0.0001, 'dense_l2_reg_c': 3e-05, 'metrics_d': 1.1}
```

- increased metrics_d to 1.1 because the minimium distance between any two random points in 128 dimensional hypersphere is 1.18 
- Reduced the embedding size to 128 
- when the training for another 100 epochs, it overfits


###  20220423-163147 

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-155249', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 128, 'lr': 0.0001, 'dense_l2_reg_c': 3e-05, 'metrics_d': 1.1}
```

- when finetune=true, the val and loss started from high value and the imddidately started overfitting
- after 32 epochs, val_VAL is settling to 0.6687 and may go down even more.


### 20220423-164333

```
 Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-155249', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 128, 'lr': 0.0001, 'dense_l2_reg_c': 0.0003, 'metrics_d': 1.18}

```
20220423-164333 - 'dense_l2_reg_c': 0.0003
20220423-165504 - 'dense_l2_reg_c': 0.001
20220423-165546 - 'dense_l2_reg_c': 0.01
20220423-165603 - 'dense_l2_reg_c': 0.03

All of them overfit, for 0.03 and 0.01, the initial loss function is in the order of 10s and because of the gradient momentum,  loss decreased rapidly than others.

with 0.01, val_VAL is slightly better than the others.


### 20220423-172212 

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-155249', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 128, 'lr': 0.0001, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.18}
```

Loss function changed to 

```python
triplet_loss = tf.reduce_sum(triplet_loss) / (num_valid_triplets + 1e-16)
```

- still overfits
- val_VAL is worse than before (C = 0.01).

###  20220423-175238

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-155249', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 128, 'lr': 0.0001, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.18}
```

Optimizer changed to SGD

- lr - 0.0001 - 20220423-175238
- lr - 0.001 - 20220423-180823
- lr - 0.01, decay_steps=13 with lr schedular - 20220423-184235 - with dense_l2_reg_c = 0.001
- lr - 0.1, decay_steps=13 with lr schedular -  20220423-184951 - with dense_l2_reg_c = 0.001



### 20220423-192535 

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=40, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 16, 'lr': 0.05, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.001, 'metrics_d': 1.18}
```

- dimensions reduced to 16 and SGD
- Back to old loss function


 ### 20220423-193918 

 ```
 Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=40, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-192535', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 16, 'lr': 0.05, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.001, 'metrics_d': 1.18}
```

- Added fintunning


 ### 20220423-195038
 ```
 Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=160, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-193918', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 16, 'lr': 0.05, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.001, 'metrics_d': 0.7}

```

- continued training using previous weights
- metric_d = 0.7

###  20220423-195726

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=40, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 64, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.1}
```

- moved from SGD to Adam
- chnaged D to 64 and metric_d =1.1
- without fine tune

###  20220423-200539 
```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-195726', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 64, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.1}

```

- continued previous training using previous weights
- val_VAL is worst that for D=128

###  20220423-201501 

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=40, finetune=False, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest=False, save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 256, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.25}
```
- D is 256
- without finetunning


###  20220423-202421 

```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-201501', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 256, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.25}
```

- Continued training on pevious weights
- D=256
-Val_VAL =0.78 is the highest than all. 2dn best is with D=128


###  20220423-223813
```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-223246', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 256, 'use_avg_pool': False, 'dropout1_rate': 0.3, 'dropout2_rate': 0.3, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.25, 'triplet_strategy': 'batch_all'}
```
- D=256 with dropout of 0.3

###  20220423-225016 
```
Namespace(data_dir='/N/u/deduggi/Carbonate/ELPephant-cropped', epochs=200, finetune=True, log_dir='logs/', params='hyperparameters/initial_run.json', restore_best=False, restore_latest='latest_weights/20220423-223246', save_freq=20)
{'image_size': 256, 'batch_size': {'train': 128, 'val': 512}, 'margin': 0.5, 'squared': False, 'embedding_size': 256, 'use_avg_pool': False, 'dropout1_rate': 0.8, 'dropout2_rate': 0.3, 'lr': 0.0001, 'decay_steps': 13, 'decay_rate': 0.96, 'dense_l2_reg_c': 0.01, 'metrics_d': 1.25, 'triplet_strategy': 'batch_all'}
```
- D=256 with dropout of 0.8




### 20220424-181645 - 20220424-183102 - 20220424-184449

Previously in data.py, images are cached after data augmentation is applied