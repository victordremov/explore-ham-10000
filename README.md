# Exploring HAM 10000 Skin Cancer Dataset
2019.07.13 - 2019.07.14

## Report structure
The report is focused on
1) getting initial dataset and skin cancer diagnostics understanding
1) planning product development process that allows short iteration-feedback loop
1) prototyping base solution to have enough data to discuss with dermatologists

In my experience, understanding strategic goals and developing step by step approach on how to reach them gives much more results than concentrating on model building as a task in itself. I see modeling as a part of product development cycle: business goals - data gathering - modeling - evaluation - production. Modeling is very much dependant on such factors as: who will be users of the model -patients or doctors, what budget we have in the company, what is the optimal balance between quality of improved diagnostics and its cost, application methods - online or embedded into diagnostics systems, etc 

To sum up, it is really important to specify the problem itself as this heavily impacts data preparation, modeling, evaluation and deployment.

## Getting initial data understanding
### First look at the data allowed for the conclusions below:
- 7470 pigmented lesions with 1-6 images each, 10015 lesion images in total
- Each lesion described with lesion's localization, patient's age and sex, and how the diagnosis was obtained
- 7 different diagnosis, highly unbalanced class size
- Lesion classes distributions among specific sex significally differs from lesion class distribution among all patients. For instance, [TODO paste example]. It means that patient's sex is a valuable feature for diagnostics. The same holds for patient age and lesion localization.

### Diving into description paper (TODO add link)
- Blurry and insufficiently scaled images, non-pigmented lesions are removed from the dataset
- Dataset is thoroughly verified: Each diagnosis has been verified for plausibility by two specialists independently and lesions with implausible diagnoses are removed.
- Treatment for various diagnoses varies from doing nothing or follow-up to taking biopsy and excision the lesion as soon as possible. It means that 1) correct diagnoses save much work, time, money and is beneficial for patient's health and 2) impact level of wrong diagnosis varies very heavily depending on what is the true lesion type and what type is predicted.
- About 95% of pigmented lesions types are covered by the dataset while more rare types are excluded.
- Such features of lesions as: structure symmetry, color and quantity of colored inclusions, are important for correct diagnostics. For better problem solving, our model should extract these features. 

## Basic business understanding
This chapter contains only generic ideas. The list should be complemented after interviewing doctors and patients. Existing items should be specified after interviewing the doctors, obtaining skin cancer diagnostics details and statistics and getting feedback from experts.

### Generic problems in skin cancer diagnostics and probable solutions
Problem | Solution | Application | Effect
--- | --- | --- | ---
Treatment cost is high | Build a model that detects a specific malignant lesion type with same sensitivity (true positive rate) as human experts but higher specificity (true negative rate)| Use model as a "second opinion" tool for dermatologists | Less benign lesions are to be biopsyed or excised leading to lower treatment and after-treatment costs. Also, lower impact on health of patients with benign lesions
Not enough dermatologists, patients have to wait for diagnosis | Build a model that classifies malignant lesions over benign ones with very high specificity | Images are taken by lower medical personal, processed on the remote server at the time of осмотр, patients with easiest benign lesions do not further осматриваются, other are routed to specialists | Part of the patients are handled by lower medical personal, the time of specialists is used more effectively, patients spend less time waiting for diagnostics
Dermatologists are unavailable e.g. in rural areas | same as previous | same as previous | Basic diagnostics is accessible even in rural areas
Patients want to verify their diagnosis | Find out cost and impact on health of misclassification for each (true class, predicted class) pair and train model that performs comparable to human experts | Make an online tool to classify dermascopic images | Less malpractice and a strong marketing tool

## Base solution prototyping
This chapter aims to build a prototype and describe it in a way that is can be interpreted by doctors. No fine-tuning here, it is too early for this - problem to be solve is yet to be chosen and evaluation metric is tightly connected with it. So the goal is to build a model to show on the first interview with a dermatologist and focus on something that is there and make discussion more product-oriented.

### Explore different model architectures / hyper parameter settings and compare their performance.
So first thing I did was group rows containing same lesion into one row with image ids combined into list. This remove bias made by duplicated sex, age and lesion localization.
[TODO example]
Interpretable metrics:
- classification report (for interview)
- sensitivity and specificity for each class (for interview)
More sensitive (but sadly not interpretable) approximations:
- mean difference between predicted probability of the true class and predicted probability of the predicted class
All models optimize cross-entropy internally, so it will also be included in statistics.
- cross-entropy

1. Logistic regression on one-hot-encoded sex, age and lesion's localization. Images are ignored.

    Metrics values:

2. Use frosen pretrained DenseNet201 to extract features from images, train logistic regression upon these features. Sex, age and lesions's localization are ignored.
    
    Metrics values:

3. Finetuning pretrained DenseNet201 with last fully-connected layer changed to match actual class count. Sex, age and lesions's localization are ignored.
 
    Metrics values:

4. Use all features: features are extracted from with pretrained DenseNet201 with last fully-connected layer removed, concatenate them with one-hot-encoded sex, age and lesion's localization and stack a fully connected layer upon them.

    Metrics values:

### The data set is quite small, what kind of implications does it have on your choices to solve the problem?
1) Image feature extractors are underfitted because input is not very diverse. This is overcomed by using transfer learning - CNN initialized with a good initial point trains faster and usually performs better. Also image augmenting increase input diversity. In this work images augmented with random rescaling, rotation and reflection to imitate making image from different angle and with different magnifying.
2) Add data from other datasets (not done in this report)
3) When comparing models with very close performance, sometimes we have to use k-fold validation or even repeated k-fold validation to lower p-value and confidence interval for metrics (not done in this report).

### Select and describe a suitable set of training techniques.
- data augmentation: images are randomly rotated, cropped and resized. Sex, and location are one hot encoded. Age is not one hot encoded but instead encoded as several 0/1 columns: (age <= 0), (age <= 5), (age <= 10) and so on to presume age order. Missing values are filled with column mean.
- store model at iteration that gives best result on validation set
- decrease learning rate reaching plateau

### Suggest a performance metric for the model and explain its clinical relevance
- sensitivity and specificity for each class evaluated on validation data
These two metrics allows us to compare human experts with the model and also to tune class weights to transform logits to class label.
- classification report evaluated on validation data: this is also very valueable to find out how exactly the model make mistakes. Each non-diagonal item negatively impacts patient's health, spends time of patient and doctors and increase treatment cost. For example, if melanocytic nevi is misclassified as melanoma - it will be excited. The impact on health varies from 14 days of post-treatment to death. Another story is when melanocytic nevi is misclassified as dermatofibroma - both are benign so there is no impact on health.

Having this losses estimated and [сведены] in same scale, we can evaluate our models directly with business metrics. This greatly helps us formulate optimization problem and directly optimize what business needs.

### Specify a verification method for the final model and verify the model. You can split the dataset as you wish but motivate and explain your choices.
Group different images of same lesion and then cross-validate with stratified 3-fold. Validation images are augmented the same way as training set.
Stratification is used to fight class imbalance, and there only 3 folds to increase training speed.

TODO: statistical significance of the verification results.

### Visualize the results and the learning procedure.
TODO: paste images
