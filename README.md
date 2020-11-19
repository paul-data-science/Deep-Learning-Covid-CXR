# Deep Learning Explainability of How Model Detected Anomalies on Covid Chest X-rays
NJIT CS677 Deep Learning Project Explainability of Covid Chest Xrays

They give you [the COVID X-ray / CT Imaging dataset](https://github.com/ieee8023/covid-chestxray-dataset) and:

1. First you find this [this implementation](https://github.com/aildnont/covid-cxr) of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read [this article](https://towardsdatascience.com/investigation-of-explainable-predictions-of-covid-19-infection-from-chest-x-rays-with-machine-cb370f46af1d) and you get your hands dirty and replicate the results in your colab notebook with GPU enabled kernel:
    - For this Part 1 task please refer to the following Jupyter Notebook files:
    </br>**_*All Image outputs are presented on their respective Jupyter Notebook files. Please scroll down all the way to the bottom to see the images._**
    </br>**_*We used DCNN RESNET model as specified from the original model.py file [https://github.com/aildnont/covid-cxr/blob/master/src/models/models.py](https://github.com/aildnont/covid-cxr/blob/master/src/models/models.py)_**
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/gradcam.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/gradcam.ipynb)
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/lime\_explain.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/lime_explain.ipynb)
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/predict-Kaggle-Fig1-Covid-Images.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/predict-Kaggle-Fig1-Covid-Images.ipynb)
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/predict.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/predict.ipynb) _*NOTE: Maybe same as above 'predict-' file_
        - **_Best performing trained model using random oversampling:_**
       </br>-[https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/BEST-SCORE-train-random\_oversample-200-epochs-Mon1am-11092020.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/BEST-SCORE-train-random_oversample-200-epochs-Mon1am-11092020.ipynb)
        - **_Worst performing trained model using class weights:_** 
       </br>[https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/BAD-SCORE-train-Class-Weights-200-epochs.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/BAD-SCORE-train-Class-Weights-200-epochs.ipynb)
       - https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/preprocess.ipynb
       
      </br>**_Note that there were code errors from the original authored train.py:_**
      </br> **Error (1) AttributeError: &#39;list&#39; object has no attribute &#39;keys&#39; Class weights line of code:**
                
                        >   if class_multiplier is not None:
                        >       class_weight = [class_weight[i] * class_multiplier[i] for i in range(len(histogram))]           
      </br> **Needed to convert to dictionary notation:**
                    
                        >   if class_multiplier is not None:
                        >       class_weight =  {i: class_weight[i] * class_multiplier[i] for i in range(len(histogram))}
      </br> **Error (2) &#39;accuracy&#39; is stated twice:**
        
                        >   metrics = [ 'accuracy', CategoricalAccuracy(name='accuracy'), ...
      </br> **Needed to remove &#39;accuracy&#39; from metrics list.**
         
                        >   metrics = [CategoricalAccuracy(name='accuracy'), ...
         

2. A fellow AI engineer, tells you about another method called [SHAP](https://arxiv.org/abs/1705.07874) that stands for SHapley Additive exPlanations and she mentions that Shapley was a Nobel prize winner so it must be important. You then find out that [Google is using it and wrote a readable white paper](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf) about it and your excitement grows. Your manager sees you on the corridor and mentions that your work is needed soon. You are keen to impress her and start writing your  **3-5 page**  summary of the SHAP approach as can be applied to explaining deep learning classifiers such as the ResNet network used in (1).
3. After your presentation, your manager is clearly impressed with the depth of the SHAP approach and asks for some results for explaining the COVID-19 diagnoses via it. You notice that the extremely popular [SHAP Github repo](https://github.com/slundberg/shap) already has an example with VGG16 network applied to ImageNet. You think it wont be too difficult to plugin the model you trained in (1) and explain it:

    - For this Part3 task our best performing saved model with random oversampling was used to run SHAP Gradient Explainer demo. </br>Please refer to [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/PaulAg-GradientExplainer-Covid-Example-model20201109-115150-h5-Pred-Set.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/src/PaulAg-GradientExplainer-Covid-Example-model20201109-115150-h5-Pred-Set.ipynb)
    </br>**_*All Image outputs are presented on their respective Jupyter Notebook files. Please scroll down all the way to the bottom to see the images._**

    - Packages pip installed on Windows 10 Anaconda (64bit Latest Download at the time):
      - Python==3.6.9 {reduced from version 3.8}
      - Jupyter== {Latest Version)
      - Keras==2.4.3
      - Tensorflow==2.3.1
      - NOTE: First create separate environment then try &quot;conda install&quot; to install packages on console environment. If &quot;conda install&quot; does not go thru then force the package install using &quot;pip install&quot;.
