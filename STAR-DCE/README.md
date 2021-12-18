## STAR-DCE

Implementation for paper "STAR: A Structure-aware Lightweight Transformer for Real-time Image Enhancement" on low light enhancement tasks.



To train the AttentionDCE model with HDRPlus paired images (256x256x3):
```
python train_attentiondce.py
```

To test the AttentionDCE model on batch2 portrait photos (1152x864x3): 
```
python test_attentiondce_fullsize.py
```
Note that the curves are estimated from 256x256x3 downsized image, then upsampled and applied to the full size inputs.

Slides (updated Sep 8, 2020):
- [link](https://docs.google.com/presentation/d/14r4SLo9fWjnomDoTmO8GJuWz4MKa_TJO04OKvZX2cHg/edit?usp=sharing)
- preprocessing pipeline
- test results on night portrait photos
- result analysis and comparison with references

