# "Training-Free Zero-Shot Temporal Action Detection with Vision-Language Models

<!--
[Michaël Defferrard](https://deff.ch),
[Kirell Benzi](https://kirellbenzi.com),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](https://www.ntu.edu.sg/home/xbresson). 
International Society for Music Information Retrieval Conference (ISMIR), 2017.\
-->

## ABSTRACT

> Existing zero-shot temporal action detection (ZS-TAD) 
> methods predominantly use fully supervised or unsuper-
> vised strategies to recognize unseen activities. However, 
> these training-based methods are susceptible to domain 
> shifts and entail high computational costs. Unlike previous 
> works, we propose a training-Free Zero-shot temporal Action 
> Detection (FreeZAD) method, leveraging image-pretrained 
> vision-language models (VLMs) to directly classify and 
> localize unseen activities within untrimmed videos. We 
> mitigate the need for explicit temporal modeling and reliance 
> on pseudo-label quality by designing the Logarithmic decay 
> weighted Outer-Inner-Contrastive Score (LogOIC) and 
> frequency-based actionness calibration. Furthermore, we 
> introduce a test-time adaptation (TTA) strategy using 
> Prototype-Centric Sampling (PCS) to expand FreeZAD, enabling 
> VLMs to adapt more effectively for ZSTAD. Extensive experiments 
> on the THUMOS14 and ActivityNet-1.3 datasets demonstrate that
> our training-free method outperforms state-of-the-art unsupervised 
> methods while requiring only 1/13 of the runtime. When equipped 
> with TTA, the enhanced method further narrows the gap with fully 
> supervised training methods of ZSTAD.

* Paper: [`arXiv:2501.13795`](https://arxiv.org/abs/2501.13795)

## Dataset

Offline features can accelerate inference. You can extract features from the [THUMOS14] dataset using [CoCa] as follows:
```bash
python extract_features.py --save_dir ./output --video_anno_path ./meta/thumos_annotations.json --video_path_file ./videos
```
[THUMOS14]:https://www.crcv.ucf.edu/THUMOS14/home.html
[CoCa]: https://arxiv.org/abs/2205.01917

## Results

The results on the THUMOS dataset using the 75%-25% split with split strategy 0 can be obtained by running:
```bash
python evaluate.py
```
The implementation details will be released upon acceptance.


## Acknowledgments and Licenses

Please cite our work if you use our code or data.

```
@article{han2025training,
  title={Training-Free Zero-Shot Temporal Action Detection with Vision-Language Models},
  author={Han, Chaolei and Wang, Hongsong and Kuang, Jidong and Zhang, Lei and Gui, Jie},
  journal={arXiv preprint arXiv:2501.13795},
  year={2025}
}
```

* The code in this repository is released under the [MIT license](LICENSE.txt).

