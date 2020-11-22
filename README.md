# Deep Learning Explainability of How Model Detected Anomalies on Covid Chest X-rays
CS 6771J1, Deep Learning, Fall 2020</br>
New Jersey Institute of Technology</br>
Project 2 - Deep Learning Project Explainability of Covid Chest Xrays</br>
Group 4 Participants:</br>
Paul Aggarwal</br>
Navneet Kala</br>
Akash Shrivastava</br>

They give you [the COVID X-ray / CT Imaging dataset](https://github.com/ieee8023/covid-chestxray-dataset) and:

1. First you find this [this implementation](https://github.com/aildnont/covid-cxr) of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read [this article](https://towardsdatascience.com/investigation-of-explainable-predictions-of-covid-19-infection-from-chest-x-rays-with-machine-cb370f46af1d) and you get your hands dirty and replicate the results:
    </br>**_*All Image outputs are presented on their respective Jupyter Notebook files._</br>_*Please scroll down all the way to the bottom of each file to see the images._</br>**_*We used DCNN RESNET model as specified from the [original model.py file](https://github.com/aildnont/covid-cxr/blob/master/src/models/models.py)_
    - **For Part (1) task please refer to the steps listed below with their respective Jupyter Notebook files:**
    
        **Step 1: Prepocess**
        - https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/preprocess.ipynb
        
        </br>**Step 2: Train (Did both Class Weights and Random Oversample)**
        - **_Trained model using class weights (Did not perform as well as random oversample):_** 
       </br>[https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/BAD-SCORE-train-Class-Weights-200-epochs.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/BAD-SCORE-train-Class-Weights-200-epochs.ipynb)
         - **_Trained model using random oversample (Best performing model):_**
       </br>[https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/BEST-SCORE-train-random\_oversample-200-epochs-Mon1am-11092020.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/BEST-SCORE-train-random_oversample-200-epochs-Mon1am-11092020.ipynb)
       
        </br>**Step 3: Predictions (Used Kaggle Fig1 Covid Images for testing predictions)**
        </br>
        </br>**_*All Image outputs are presented on their respective Jupyter Notebook files._**
        </br>**_*Please scroll down all the way to the bottom of the file to see the images._**
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/predict-Kaggle-Fig1-Covid-Images.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/predict-Kaggle-Fig1-Covid-Images.ipynb)
        
        </br>**Step 4: Lime Explainer**
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/lime\_explain.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/lime_explain.ipynb)
        
        </br>**Step 5: Grad-Cam Explainer**
        - [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/gradcam.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-1/src/gradcam.ipynb)
        
        
        
       
        
       
       
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
         
    - Important packages pip installed on Windows 10 with NVIDIA GTX 1650 GPU (NVIDIA CUDA 11.1 and cuDNN 8.0.4.30 drivers) on Anaconda (64bit Latest Download at the time):
      - Python==3.7.9 {reduced from version 3.8}
      - Jupyter== {Latest Version)
      - Keras_gpu==2.3.1
      - Tensorflow_gpu==2.3.1
      - NOTE: First create separate environment then try &quot;conda install&quot; to install packages on console environment. If &quot;conda install&quot; does not go thru then force the package install using &quot;pip install&quot;.

    - **To replicate this Part (1) yourself**, you need to pull from https://github.com/aildnont/covid-cxr and download or pull the contents of our [Part-1 folder](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/tree/main/Part-1) into your './covid-cxr-master' folder. You will also have to make file path changes to the 'config.yml' file under the './covid-cxr-master' folder. Then go directly to './covid-cxr-master/src' folder to run the Jupyter notebooks using the steps outlined above in the Part (1) task.
      <br/><br/>

2. A fellow AI engineer, tells you about another method called [SHAP](https://arxiv.org/abs/1705.07874) that stands for SHapley Additive exPlanations and she mentions that Shapley was a Nobel prize winner so it must be important. You then find out that [Google is using it and wrote a readable white paper](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf) about it and your excitement grows. Your manager sees you on the corridor and mentions that your work is needed soon. You are keen to impress her and start writing your  **3-5 page**  summary of the SHAP approach as can be applied to explaining deep learning classifiers such as the ResNet network used in (1):
    - **For Part (2) please refer to** [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-2/Part2_SHAP_Explainability.pdf](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-2/Part2_SHAP_Explainability.pdf)
    
3. After your presentation, your manager is clearly impressed with the depth of the SHAP approach and asks for some results for explaining the COVID-19 diagnoses via it. You notice that the extremely popular [SHAP Github repo](https://github.com/slundberg/shap) already has an example with VGG16 network applied to ImageNet. You think it wont be too difficult to plugin the model you trained in (1) and explain it:

    - **For Part (3) task our best performing saved model from Part (1) with random oversample was used to run SHAP Gradient Explainer demo.**
    </br>**Please refer to** [https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-3/PaulAg-GradientExplainer-Covid-Example-model20201109-115150-h5-Pred-Set.ipynb](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-3/PaulAg-GradientExplainer-Covid-Example-model20201109-115150-h5-Pred-Set.ipynb)
    
    - Important packages pip installed on Windows 10 on Anaconda (64bit Latest Download at the time):
      - Python==3.6.9 {reduced from version 3.8}
      - Jupyter== {Latest Version)
      - Keras==2.4.3
      - Tensorflow==2.3.1
      - NOTE: First create separate environment then try &quot;conda install&quot; to install packages on console environment. If &quot;conda install&quot; does not go thru then force the package install using &quot;pip install&quot;.
      
    - **To replicate this Part (3) yourself**, you need to pull from https://github.com/slundberg/shap and download or pull our [Jupyter notebook example of Gradient Explainer](https://github.com/paul-data-science/Deep-Learning-Covid-CXR/blob/main/Part-3/PaulAg-GradientExplainer-Covid-Example-model20201109-115150-h5-Pred-Set.ipynb) into your 'shap-master' directory. Then also copy and paste your saved model.h5 (which you may have generated in the "Part (1) task - Step 2: Train" section) to the same directory. Set your images path directory, your model.h5 and 'predictions.csv' inside the Jupyter notebook.
