# Characters Recognition

A Chinese characters recognition repository based on convolutional recurrent networks. (**Below please scan the QR code to join the wechat group.**)

## Train
```angular2html
   [run] python train.py --cfg lib/config/360CC_config.yaml
or [run] python train.py --cfg lib/config/OWN_config.yaml
or [run] python train.py --cfg lib/config/digit_config.yaml
or [run] python train.py --cfg lib/config/hanzi_config.yaml
```
```
#### loss curve

```angular2html
   [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
   [run] tensorboard --logdir log
```

#### loss overview(first epoch)
<center/>
<img src='images/train_loss.png' title='loss1' style='max-width:800px'></img>
</center>
<p>
<img src='images/tb_loss.png' title='loss1' style='max-width:600px'></img>
</p>

## Demo
```angular2html
   [run] python demo.py --cfg lib/config/digit_config.yaml --image_path images/10500.png --checkpoint output/DIGIT/crnn/checkpoints/best__09ep__0.9700acc__val_checkpoint.pth.tar
```
## References
- https://github.com/meijieru/crnn.pytorch
- https://github.com/HRNet




