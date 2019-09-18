```                
 _______     ___  ____   ____  _____  ____  _____  
|_   __ \   |_  ||_  _| |_   \|_   _||_   \|_   _| 
  | |__) |    | |_/ /     |   \ | |    |   \ | |   
  |  __ /     |  __'.     | |\ \| |    | |\ \| |   
 _| |  \ \_  _| |  \ \_  _| |_\   |_  _| |_\   |_  
|____| |___||____||____||_____|\____||_____|\____|     
```
## Usage
1. Transform caffe model to rknn model in PC
```
./transfer.py
```
2. Predict result in RK3399Pro
```
./predict.py
```


### Announcements
1. rknn-toolkit require version 1.1.0 when transfer model, because version 1.2.0 has bugs for quantization
