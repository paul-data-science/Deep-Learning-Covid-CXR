# Deep-Learning-Covid-CXR
NJIT CS677 Deep Learning Project Explainability of Covid Chest Xrays

They give you [the COVID X-ray / CT Imaging dataset](https://github.com/ieee8023/covid-chestxray-dataset) and:

1. First you find this [this implementation](https://github.com/aildnont/covid-cxr) of the method called Local Interpretable Model-Agnostic Explanations (i.e. LIME). You also read [this article](https://towardsdatascience.com/investigation-of-explainable-predictions-of-covid-19-infection-from-chest-x-rays-with-machine-cb370f46af1d) and you get your hands dirty and replicate the results in your colab notebook with GPU enabled kernel.
2. A fellow AI engineer, tells you about another method called [SHAP](https://arxiv.org/abs/1705.07874) that stands for SHapley Additive exPlanations and she mentions that Shapley was a Nobel prize winner so it must be important. You then find out that [Google is using it and wrote a readable white paper](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf) about it and your excitement grows. Your manager sees you on the corridor and mentions that your work is needed soon. You are keen to impress her and start writing your  **3-5 page**  summary of the SHAP approach as can be applied to explaining deep learning classifiers such as the ResNet network used in (1).
3. After your presentation, your manager is clearly impressed with the depth of the SHAP approach and asks for some results for explaining the COVID-19 diagnoses via it. You notice that the extremely popular [SHAP Github repo](https://github.com/slundberg/shap) already has an example with VGG16 network applied to ImageNet. You think it wont be too difficult to plugin the model you trained in (1) and explain it.
