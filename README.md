# MADAIN
## Mole Analysis with Deep Adam-optimized Inception Network

### By: Amanda Derdiger, Andrew Koller, Mustafa Can Ayter, and Natalia Mitchell

### Introduction
We have built a convolutional neural network (CNN) to analyze images of skin lesions and categorize them into one of seven classes, three of which are cancerous and four of which are benign. We have have also developed a web page, currently hosted on GitHub pages, and plan to embed a web app with our model.

### Data
Our dataset is from Kaggle and can be accessed by the link below. This dataset contains 10,015 images of skin lesions across the 7 classes detailed below.

Demographics:

![demographics](https://github.com/aderdiger/MADAIN/assets/148494444/b67fe672-58c2-47b1-9d63-bf7c1cfeb654)

Class Definitions:

![image](https://github.com/aderdiger/MADAIN/assets/148494444/e6325a45-a732-4b5b-a062-8a81e86cea94)

https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset

### Process

#### Benchmarking
We started by benchmarking three CNN architectures detailed in Aurelien Geron's book "Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow" (2022). These three CNNs were: InceptionV3, ResNet50, and VGG16. In addition to the three architectures, we tested 3 different optimizers for each: Adam, RMSprop, and SGD. In total, nine models were benchmarked, and from those, we chose InceptionV3 with the Adam optimizer as our primary model.*

* For benchmarking metrics, see "run1/visualizations/"

Despite the InceptionV3.Adam slightly underperforming relative to ResNet50.Adam in the classification reports, InceptionV3.Adam was chosen for it's supperior performance on AUC metrics. (see \run1\visualizations\roc_curve\roc_curve_InceptionV3_Adam.png). 

#### Fine-Tuning the Model
Once we chose our primary model, we continued to fine-tune it to maximize our AUC, precision, and recall scores, with recall on our three cancerous classes more highly prioritized. This is because, in the precision/recall trade-off, favoring recall reduces false negatives. In a cancer identification model, such as this, false negatives in the cancerous classes would be our most detrimental outcome that should be minimized to the extent possible. Please note the 'Cancer Catcher' model in run4, which reached our higest recall for melanoma at .7. 

Our fine-tuning steps, along with their corresponding run folders in our repo are detailed below.

1. Running InceptionV3.adam at 150 epochs (run3; v6)
    1. Removed image augmentation - original benchmarking involved preliminary image augmentation:
        
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,

    2. Results from testing do not show improvement with higher epochs. 

    ![image](https://github.com/aderdiger/MADAIN/blob/main/run3/visualizations/roc_curve_InceptionV3_Adam.png)

               precision    recall  f1-score   support

       akiec       0.02      0.05      0.03        65
         bcc       0.05      0.09      0.07       103
         bkl       0.08      0.07      0.08       220
          df       0.00      0.00      0.00        23
         mel       0.11      0.17      0.13       223
          nv       0.68      0.50      0.57      1341
        vasc       0.01      0.04      0.02        28
   
       accuracy                               0.37      2003
       macro avg          0.14      0.13      0.13      2003
       weighted avg       0.48      0.37      0.41      2003


    3. Resolution: test at lower epochs; weights need to be adjusted 

2. Weighting scheme testing 1 - The Cancer Catcher (run4; v7)
    1. All testing to this point utilized TensorFlow's 'balanced' weighting system to account for large imblanace in classes.
    2. 4x on 'bcc' and 'akeic', 20x on 'mel'
    3. Tested effectiveness of different weights 
    4. Increased weighting for underrepresented classes by a factor of 4x
    5. Note the recall for 'mel' at .70

    ![image](https://github.com/AEKoller/MADAIN/blob/main/run4%20-%20Cancer%20Catcher/visualizations/roc_curve_InceptionV3_Adam.png)
    

                        precision    recall  f1-score   support

              akiec       0.04      0.06      0.05        65
                bcc       0.06      0.07      0.06       103
                bkl       0.00      0.00      0.00       220
                df        0.00      0.00      0.00        23
                mel       0.11      0.70      0.19       223
                nv        0.63      0.12      0.20      1341
                vasc      0.02      0.04      0.03        28
   
       accuracy                               0.16      2003
       macro avg          0.12      0.14      0.08      2003
       weighted avg       0.44      0.16      0.16      2003

4. Binary classification testing (run5; v8)
    1. Testing conducted at same time as weighting scheme testing
    2. All testing to this point involved a multiclass classifier.
    3. Tested to effectiveness of a binary classifier as opposed to a multiclass classifier.
    4. Results unremarkable

   ![image](https://github.com/AEKoller/MADAIN/blob/main/run5/visualizations/roc_curve_InceptionV3_Adam.png)

                       precision    recall  f1-score   support

              benign      0.79      0.58      0.67      1612
           cancerous      0.17      0.37      0.24       391

       accuracy                               0.54      2003
       macro avg          0.48      0.47      0.45      2003
       weighted avg       0.67      0.54      0.59      2003


5. Inverse proportional weighting (run8)
    1. Weighted classes based on the inverse of their frequency
   
6. Class balanced loss approach weighting (run9)
    1. Attempted to implement balanced loss weighting, model performed poorly

7. Adding generated augmented images to training data (run10)
    1. Added a random imgage augementor and image generator 
    2. Added randomly generated images back into training data 
    3. Wanted to normalize percentage representation in data set of underrepresented classes

    ![image](https://github.com/AEKoller/MADAIN/blob/main/run10/visualizations/roc_curve_InceptionV3_Adam.png)

                    precision    recall  f1-score   support

            akiec       0.01      0.02      0.01        65
              bcc       0.04      0.06      0.05       103
              bkl       0.07      0.03      0.04       220
               df       0.03      0.04      0.04        23
              mel       0.12      0.22      0.15       223
               nv       0.68      0.61      0.64      1341
             vasc       0.00      0.00      0.00        28

           accuracy                           0.44      2003
          macro avg       0.14      0.14      0.13      2003
       weighted avg       0.48      0.44      0.45      2003


8. Increasing custom layer neuron density from 512 to 1024 and rerunning promissing models (run11; v12)
    1. Testing Multiple models with increased neuron count
    2. Top performers are as follows: InceptionV3.Adam, ResNet50.Adam, VGG16.SGD
    3. Ultimately, InceptionV3.Adam remained the highest perfrmer
    
    InceptionV3.Adam
   
    ![image](https://github.com/AEKoller/MADAIN/blob/main/run11/visualizations/roc_curve_InceptionV3_Adam.png)

                     precision    recall  f1-score   support

            akiec       0.03      0.06      0.04        65
              bcc       0.04      0.06      0.05       103
              bkl       0.12      0.14      0.13       220
               df       0.01      0.04      0.02        23
              mel       0.09      0.15      0.12       223
               nv       0.67      0.50      0.57      1341
             vasc       0.03      0.07      0.04        28

           accuracy                           0.37      2003
          macro avg       0.14      0.15      0.14      2003
       weighted avg       0.48      0.37      0.41      2003

    ResNet50.Adam
   
    ![image](https://github.com/AEKoller/MADAIN/blob/main/run11/visualizations/roc_curve_ResNet50_Adam.png)

                    precision    recall  f1-score   support

           akiec       0.03      0.05      0.04        65
             bcc       0.04      0.06      0.05       103
             bkl       0.12      0.13      0.12       220
              df       0.00      0.00      0.00        23
             mel       0.13      0.24      0.17       223
              nv       0.69      0.52      0.59      1341
            vasc       0.07      0.11      0.09        28

            accuracy                           0.39      2003
           macro avg       0.16      0.16      0.15      2003
        weighted avg       0.49      0.39      0.43      2003

    VGG16.SGD
   
    ![image](https://github.com/AEKoller/MADAIN/blob/main/run11/visualizations/VGG16_SGD/roc_curve_VGG16_SGD.png)

                    precision    recall  f1-score   support

           akiec       0.04      0.06      0.05        65
             bcc       0.02      0.03      0.02       103
             bkl       0.13      0.15      0.14       220
              df       0.00      0.00      0.00        23
             mel       0.15      0.24      0.19       223
              nv       0.68      0.54      0.61      1341
            vasc       0.03      0.04      0.03        28

            accuracy                           0.41      2003
           macro avg       0.15      0.15      0.15      2003
        weighted avg       0.49      0.41      0.44      2003

10. Augmented image generation with 1000 images for underrepresented classes with InceptionV3.Adam (run12; v11)
    1. This version was technically ran in two parts: the first generated augmented images such that minority classes would contain at least 500 images. The second run generated augmented images such that each minority class would contain at least 1000 images.
    2. 'df' performing relatively well, but vasc is not being identified at all
    3. Our theory was that the 'vasc' class was being subsumed into the other minority classes due to augmentation noise.

    ![image](https://github.com/AEKoller/MADAIN/blob/main/run12/visualizations/roc_curve_InceptionV3_Adam.png)

                    precision    recall  f1-score   support

           akiec       0.03      0.06      0.04        65
             bcc       0.08      0.13      0.09       103
             bkl       0.14      0.15      0.14       220
              df       0.00      0.00      0.00        23
             mel       0.12      0.22      0.15       223
              nv       0.67      0.50      0.58      1341
            vasc       0.00      0.00      0.00        28

            accuracy                           0.39      2003
           macro avg       0.15      0.15      0.14      2003
        weighted avg       0.48      0.39      0.42      2003

11. 
       


#### Resources:

https://towardsdatascience.com/review-inception-v4-evolved-from-googlenet-merged-with-resnet-idea-image-classification-5e8c339d18bc

https://stackoverflow.com/questions/51798784/keras-transfer-learning-on-inception-resnetv2-training-stops-in-between-becaus
