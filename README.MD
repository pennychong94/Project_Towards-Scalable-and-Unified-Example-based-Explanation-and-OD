
# Datasets and splits used for experiments for the paper 'Towards Scalable and Unified Example-based Explanation and Outlier Detection' 
link to paper https://arxiv.org/abs/2011.05577

## For LSUN experiments 
###### To prepare LSUN dataset (lsun_split folder) :
Refer to https://github.com/fyu/lsun to download the whole latest dataset. After downloading the dataset, run the script file command_to_run_data.sh (specify paths in the script) to  export 10,000 train images per class and all the given validation images as  128 x 128 PNG files.

The train, val, and test splits used in our experiments are given in the folder:  
+ *lsun_split/new_allcls_lsun_split/*
    + *new10k_train.txt*
    + *new10k_val.txt*
    + *new_test.txt*
    + *new_test_subset100.txt*  &nbsp;&nbsp;&nbsp;  (subset used in LRP perturbation)

Each line is in the above textfiles is in the form of:
+ *imagename* &nbsp; *class_label*

To create LSUN strokes and altered color outlier samples, cd to *lsun_split/codes_genoutlier_data* folder. Run the following commands:
>>python gen_synthetic_strokes.py --dataset lsun --thickness 5 --rootPath /some/path/to/LSUN_images_PL/ 
>>python gen_altered_colors.py --dataset lsun --rootPath /some/path/to/LSUN_images_PL/
 

## For PCam experiments 
###### To prepare PCam dataset (pcam_split folder) :
We used the train, val, and test  split from https://github.com/basveeling/pcam.

The tar file *pcam_split/new_allcls_patchcam_split.tar.xz* contains the following test subsets used in LRP perturbation and outlier evaluation:  
+ *camelyonpatch_level_2_split_test_subset500_x.h5*  &nbsp;&nbsp;&nbsp;      (subset used in LRP perturbation)
+ *camelyonpatch_level_2_split_test_subset500_y.h5*   &nbsp;&nbsp;&nbsp;  (subset used in LRP perturbation)
+ *camelyonpatch_level_2_split_test_subset1500_x.h5*  &nbsp;&nbsp;&nbsp;      (subset used in outlier evaluation)
+ *camelyonpatch_level_2_split_test_subset1500_y.h5*   &nbsp;&nbsp;&nbsp;  (subset used in outlier evaluation)

To create PCam strokes and altered color outlier samples, cd to *pcam_split/codes_genoutlier_data* folder. Run the following commands:
>>python gen_synthetic_strokes.py --dataset patchcam --thickness 5 --rootPath /rootpath/to/new_allcls_patchcam_split/
>>python gen_altered_colors.py --dataset patchcam --rootPath /rootpath/to/new_allcls_patchcam_split/

## For Stanford Cars experiments 
###### To prepare Stanford dataset (lsun_split folder) :
Download the dataset from http://ai.stanford.edu/~jkrause/cars/car_dataset.html and put them in the folder *stanford_cars*. Then perform data augmentation on the images in the folder *stanford_cars/cars_train/* using the same data augmentation as the authors from the ProtoPNet work https://github.com/cfchen-duke/ProtoPNet/blob/master/img_aug.py

We used only samples from the first 50 car models. The following textfiles contain the **unaugmented** train, val, and test splits: 
+ *stanfordcar_split/new_allcls_stanfordcar_split/*
    + *new_train90_50classes_withoutdataug.txt*  &nbsp;&nbsp;&nbsp;  (unaugmented training images)
    + *new_val10_50classes.txt*
    + *test_50classes.txt*

For training, we used the **augmented** images of the original image in *new_train90_50classes_withoutdataug.txt* and thus we have 1835 x 30 = 55,050 training samples in total. For validation and test, we used the **unaugmented** images as listed in *new_val10_50classes.txt* and *test_50classes.txt*, respectively.

To create Stanford Cars strokes and altered color outlier samples, cd to *stanfordcar_split/codes_genoutlier_data* folder. Run the following commands:
>>python gen_synthetic_strokes.py --dataset stanfordcar --thickness 5 --rootPath /path/to/stanford_cars/
>>python gen_altered_colors.py --dataset stanfordcar --rootPath /path/to/stanford_cars/

