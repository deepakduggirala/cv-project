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