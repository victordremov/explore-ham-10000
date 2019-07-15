# Exloring HAM10000


## State the problem to solve
 
### Let us start with naming the problems in skin cancer diagnostics

- diagnostics cost much
    - it is pricey for patients 
    - it is pricey for tax-payers
  
    It gives: 
    - not every customer can afford for skin cancer diagnostics regularly
    - tax payers money cannot be used on other spheres

  
- patient have to wait much time since application on diagnostics until final result
    - there are queues for dermatogists because there are not enough specialist
    - there queues for the picture to be processed by dermatologist
    - sometimes additional diagnostics is required, and it means more queues 
  
  In case of maligni&&& lesion the cost of healing grows higher and there are more negative impact on patient's health.

- malpractices
    - in case when benigh lesion is identified as malignant or additional diagnostics needed:
        - if it is possible to use other type of non-invasive diagnostics:
            - it costs money
            - there are queues
        - if non-invasive diagnostics is not available, the lesion is biopsied and excited
            - this costs much more
            - the method is invasive
            - the operation is painfull
            - up to 14 days to recover after the operation
            - after operation patient should be regularly observed by a doctor
    - in case malignant lesion treated as benigh.
      - longer time until treatment
        - it makes treatment more expensive
        - in case of melanoma not only money and health are on the stake - patient's lifetime could be greatly reduced 

- But really, in my opinion it is not very efficient to guess what problems are important and what are not.
These problems affect dermatologists and their patients so only they can say for sure if the problem exists and is common.
So talking and collecting feedback from dermatologists, health department workers and patients are the key to good product.   

### How an ML model could be of help

- Make diagnostics cheaper
    
    If we have an ML model with performance comparable with dermatologist's performance or a little lower, we can use this model as a filter for patients.
    All pictures are analysed and most obvious benigh cases are not sent to oncologist unless patient explicitly ask to do this.
    This would make treatment cheaper, patients would get feedback faster.
    This lowers patients's torrent and dermatologists can use more time for more difficult cases.
    
    For example, we can integrate a model with dermascopes' software. Thus an operator can get results on site, without need to get consilium.


- Make diagnostics faster and more available for the patients.
    
    For example, we can make web-service for patients. Web service analize uploaded pictures and in case of подозрений на malignant lesion motivate the patient to go to doctor..
    - it lowers требования for dermatologists who operate dermascope - it is much easier to make a picture then to analize it.
    - patients can get "second opinion" for free to double check the diagnosis they get from the doctor.
    - as an additional feature, such a website is a great promotion tool for the company and its other services.


- Improve diagnostics accuracy by using model as a tool for dermatologist.
    
    Precise diagnosis saves patient's spent time and lowers negative impact caused by invasive treatment methods of benigh lesions.
    
    Using model as tie breaker increases diagnosis accuracy and results which is better for the customers.  
    This is a higher level product and patients can pay more for it to save health and time.
 

## Explore different model architectures / hyper parameter settings and compare their performance.
   - encode patient's sex, age and lesion's localization
- unite them with features extracted from picture using pretrained deep learning models from `torch.model_zoo` e.g. Inception, ResNet, ConvNet 
- stack dense layers with feature union as input as lesion classes as output
- to improve train speed, first layers of pretrained models are frozen 


## The data set is quite small, what kind of implications does it have on your choices to solve the problem?
Значение функции потерь по кросс-валидации зависят от разбиения на обучающую и тестовую выборку.
Дисперсия функции потерь обратно пропорциональна корню из размера тестовой выборки и обратно пропорциональна корню из количества разбиений на обучающую и тестовую выборку.
Поэтому для выборок маленького размера мы вынуждены использовать k-fold и tk-fold для более точного измерения метрик и настройки гиперпараметров.  

Вторая сложность - модель легко переобучается на малых данных.
Чтобы уменьшить импакт мы используем аугментацию для увеличения разнообразия выборки.
Также мы используем transfer learning, чтобы начать ближе к глобальному минимуму, и следовательно сойтись в локальный минимум с меньшим значением, чем при рандомной инициализации модели.  

## Select and describe a suitable set of training techniques.
Due to small time for research, I used most common technics:

- data augmentation: images are randomly rotated, cropped, resized and projected. This simulates making a picture from different angle and with different увеличение. 
- choose iteration that gives best results on validation
- decrease learning rate reaching plateau

## Suggest a performance metric for the model and explain its clinical relevance
We approximate negative impact on health, time and money if patient trusts model.
This is ultimately is a weighted sum of (1 - sensitivity) and (1 - specificity) of model on each lesion class and the higher negative impact when patient trust model the higher are weights on corresponding term.

Also should measure sensitivity and specificity for each class because this metrics are very common in medical society and can be easier compared with state-of-art.    

## Specify a verification method for the final model and verify the model. You can split the dataset as you wish but motivate and explain your choices.
Grouped k-fold splitting with test data augmented. 
We use grouped splitting to be sure no pictures of the same lesion is in both train and test set.
K-fold validation and test data augmenting make metric value more precise

## Comment on the statistical significance of the verification results.
95% confidence interval for mean negative impact:
95% confidence interval for sensitivity:
95% confidence interval for specificity: 

## Visualize the results and the learning procedure.
