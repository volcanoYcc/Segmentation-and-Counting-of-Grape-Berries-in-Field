# Segmentation and Counting of Grape Berries in Field
This repository is the official implementation of our paper: [Segmentation and Counting of Grape Berries in Field](https://temp)  

## Chengdu grape berry dataset
Chengdu dataset was captured by us with an iPhone 12 smartphone, an iPhone 13 smartphone, and an HUAWEI mate 40 pro smartphone in vineyards in Longquanyi and Shuangliu districts, Chengdu, China, in July 2023. It contains a total of 150 RGB images of three grape varieties: Kyoho, Shine Muscat and Summer Black. All the images are captured in a frontal pose with approximately the same distance to the grape vines. We add instance segmentation annotations for a total of 50718 grape berries in these images in COCO format. The original size of images and annotations are 4032×3024 and 4096×3072, and are resized to 2048×1536 in model training and testing.
Our dataset will be publically available after the acceptance of our paper.

Kyoho

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/Kyoho_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/Kyoho_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/Kyoho_3.jpg" width="260px" />
<details>
<summary>click to show more</summary>
  
Shine Muscat
  
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/ShineMuscat_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/ShineMuscat_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/ShineMuscat_3.jpg" width="260px" />

Summer Black

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/SummerBlack_1.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/SummerBlack_2.jpg" width="260px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/SummerBlack_3.jpg" width="260px" />
</details>

## Mask Gaussian kernels
Example probability maps generated using our proposed method of creating Gaussian kernels with object instance segmentation annotations.

<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/Kyoho_30.jpg" width="400px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/Kyoho_30_mask.jpg" width="400px" />
<details>
<summary>click to show more</summary>
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/ShineMuscat_13.jpg" width="400px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/ShineMuscat_13_mask.jpg" width="400px" />
<img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/SummerBlack_8.jpg" width="400px" /> <img src="https://github.com/volcanoYcc/Segmentation-and-Counting-of-Grape-Berries-in-Field/raw/master/README_images/SummerBlack_8_mask.jpg" width="400px" />
</details>

## Grape berry segmentation and counting
Grape berry instance segmention results obtained from the probability maps predicted be the neural network using watershed algorithm.

