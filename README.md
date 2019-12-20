# DAME WEB
**[ICCVw19] DAME WEB: DynAmic MEan with Whitening Ensemble Binarization for Landmark Retrieval without Human Annotation**

We present a **compact binary global descriptor** for landmark retrieval. It is **8 times smaller** than the baseline [GeM](https://github.com/filipradenovic/cnnimageretrieval-pytorch).


| Storage/Performance Comparison | Demo Video|
| --- | --- |
| <img src="https://github.com/vbalnt/dame-web-buffer/blob/master/storage_comparison.png" height="300"/> | <img src="https://github.com/vbalnt/dame-web-buffer/blob/master/dynp.gif" height="300"/> |




**Code Author: Tsun-Yi Yang**


**Paper Authors: Tsun-Yi Yang, Duy-Kien Nguyen, Huub Heijnen, Vassileios Balntas**


This is a demo version of the DAME WEB paper from [ICCV2019 CEFRL workshop](http://www.ee.oulu.fi/~lili/CEFRLatICCV2019.html).
It is done in the PhD research internship period (2019) at Scape Technologies for Tsun-Yi and Kien.


### PDF
[link](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CEFRL/Yang_DAME_WEB_DynAmic_MEan_with_Whitening_Ensemble_Binarization_for_Landmark_ICCVW_2019_paper.pdf)

### Download

To start with running the code, you have to download the **pre-trained model weight** and the **whitening ensemble file** first
[Google drive](https://drive.google.com/open?id=1GEGvq2OX88uHuVV1sL2rKLkE61rQSpN4)

Put the folders under the main folder of this repo and it's done.

### Demo

We provide [ipython file](https://github.com/vbalnt/dame-web-buffer/blob/master/demo/demo.ipynb) and a shell script as examples for running the code (i.e. extracting global binary descriptor from a custom image).
+ Shell script example
```
cd demo
sh run_extract.sh
```
+ An example command for calling the python:
```
python3 -m TYY_extract_DAME_WEB --gpu-id '0' --network-path '../pre-trained/Res101_DAME.pth.tar' --whitening-path '../whitening/WEB_retrieval-SfM-120k.pth' --image-size 1024 --image-path '../images/big_ban.jpg' --multi_dilation False
```


### Citation
If you use the code, please cite the following paper.
```
@inproceedings{yang2019dame,
  title={DAME WEB: DynAmic MEan with Whitening Ensemble Binarization for Landmark Retrieval without Human Annotation},
  author={Yang, Tsun-Yi and Kien Nguyen, Duy and Heijnen, Huub and Balntas, Vassileios},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```

### Reference
This work is heavily inspired by [GeM](https://github.com/filipradenovic/cnnimageretrieval-pytorch). You may refer the training part to GeM.
