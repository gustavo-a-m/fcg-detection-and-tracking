## Introdução

Aplicação desenvolvida para o trabalho de conclusão de curso de Gustavo Alves Moreira no curso de Engenharia de Computação na PUC Minas no ano de 2020.

## Requerimentos

* Python 3.6.8
* CMake
* OpenCV
* Numpy
* Sklean
* Pillow
* Tensorflow
* Imutils

## Para executar o projeto

- Utilizando uma ou mais imagens
```
python main.py --images path-da-imagem1 path-da-imagem2
```

- Utilizando um vídeo
```
python main.py --video path-do vídio
```

## Informações importantes

* O `fx` pode ser configurado através do argumento `--fx valor`.
* O *stride* pode ser configurado no arquivo `main.py` na declaração de um novo objeto `FFormation` (`FFormation(poses_2d_Top_View, stride=35)`).
* Configurações da câmera podem ser passadas através de um arquivo de *extrinsics* que pode ser fornecido através do argumento `--extrinsics-path caminho-do-arquivo`.
* Outras configurações podem ser alteradas, para obter uma referência utilize os respositorios apresentados na seção [Repositórios de Referência](#repositorio-de-referencias) abaixo

## Referências

```
@article{YOLO,
    author = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
    year = {2016},
    month = {06},
    pages = {779-788},
    title = {You Only Look Once: Unified, Real-Time Object Detection},
    doi = {10.1109/CVPR.2016.91}
}
```

```
@article{DeepSORT,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}
```

```
@article{GraphCuts,
    author = {Setti, Francesco and Russell, Chris and Bassetti, Chiara and Cristani, Marco},
    year = {2014},
    month = {09},
    pages = {},
    title = {F-Formation Detection: Individuating Free-Standing Conversational Groups in Images},
    volume = {10},
    journal = {PLOS ONE},
    doi = {10.1371/journal.pone.0123783}
}
```

```
@article{SingleShot,
    author = {Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Sridhar, Srinath and Pons-Moll, Gerard and Theobalt, Christian},
    year = {2018},
    month = {09},
    pages = {120-130},
    title = {Single-Shot Multi-person 3D Pose Estimation from Monocular RGB},
    doi = {10.1109/3DV.2018.00024}
}
```

## Repositórios de Referência <a name="repositorio-de-referencias"/>

https://github.com/Qidian213/deep_sort_yolov3
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch
http://vips.sci.univr.it/research/fformation/
