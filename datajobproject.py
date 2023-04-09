#!/usr/bin/env python
# coding: utf-8

# <h1>Data overview</h1>

# <h2>Importing libraries</h2>

# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# <h2>Reading file as a DataFrame</h2>

# In[3]:


df = pd.read_csv("kaggle_survey_2020_responses.csv")
df.head()


# In[4]:


df.info()


# <h2>Renaming the columns with the values of first row</h2>

# In[3]:


df.columns = df.iloc[0]
df.drop(index=0,inplace=True)
df.head()


# <h2>Renaming the columns names individually</h2>

# In[4]:


rename_columns = {
'Duration (in seconds)':'duration_seconds',
'What is your age (# years)?':'q1_age',
'What is your gender? - Selected Choice':'q2_gender',
'In which country do you currently reside?':'q3_country',
'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'q4_nxt2year_highestEdu',
'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'q5_professional_role',
'For how many years have you been writing code and/or programming?':'q6_coding_experience',
'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python':'q7_most_freq_used_prgmg_language_Python',
'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R':'q7_most_freq_used_prgmg_language_R',
'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL':'q7_most_freq_used_prgmg_language_SQL',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C':'q7_most_freq_used_prgmg_language_C',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++': 'q7_most_freq_used_prgmg_language_C++',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java':'q7_most_freq_used_prgmg_language_Java',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript':'q7_most_freq_used_prgmg_language_Javascript',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Julia': 'q7_most_freq_used_prgmg_language_Julia',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Swift': 'q7_most_freq_used_prgmg_language_Swift',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash': 'q7_most_freq_used_prgmg_language_Bash',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB': 'q7_most_freq_used_prgmg_language_MATLAB',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - None': 'q7_most_freq_used_prgmg_language_None',
 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other': 'q7_most_freq_used_prgmg_language_Other',
 'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice':'q8_recommended_prgmg_language',
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Jupyter (JupyterLab, Jupyter Notebooks, etc) ": "q9_most_freq_used_IDE_Jupyter",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  RStudio ": "q9_most_freq_used_IDE_RStudio",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code ":"q9_most_freq_used_IDE_Visual_Studio_Code",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Click to write Choice 13":"q9_most_freq_used_IDE_Choice13",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  PyCharm ":"q9_most_freq_used_IDE_PyCharm",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Spyder  ":"q9_most_freq_used_IDE_Spyder",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Notepad++  ":"q9_most_freq_used_IDE_Notepad++",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Sublime Text  ":"q9_most_freq_used_IDE_SublimeText",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Vim / Emacs  ":"q9_most_freq_used_IDE_Vim_Emacs",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  MATLAB ":"q9_most_freq_used_IDE_MATLAB",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - None":"q9_most_freq_used_IDE_None",
 "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other":"q9_most_freq_used_IDE_Other",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks':"q10_most_freq_used_notebook_product_KaggleNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Colab Notebooks':"q10_most_freq_used_notebook_product_ColabNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Azure Notebooks':"q10_most_freq_used_notebook_product_AzureNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Paperspace / Gradient ':"q10_most_freq_used_notebook_product_Paperspace_Gradient",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Binder / JupyterHub ':"q10_most_freq_used_notebook_product_Binder_JupyterHub",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Code Ocean ':"q10_most_freq_used_notebook_product_CodeOcean",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  IBM Watson Studio ':"q10_most_freq_used_notebook_product_IBMWatsonStudio",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon Sagemaker Studio ':"q10_most_freq_used_notebook_product_AmazonSagemakerStudio",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon EMR Notebooks ':"q10_most_freq_used_notebook_product_AmazonEMRNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud AI Platform Notebooks ':"q10_most_freq_used_notebook_product_GoogleCloudAIPlatformNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Datalab Notebooks':"q10_most_freq_used_notebook_product_GoogleCloudDatalabNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks Collaborative Notebooks ':"q10_most_freq_used_notebook_product_DatabricksCollaborativeNotebooks",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - None':"q10_most_freq_used_notebook_product_None",
 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':"q10_most_freq_used_notebook_product_Other",
 'What type of computing platform do you use most often for your data science projects? - Selected Choice':"q11_most_freq_used_computing_platform",
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs':'q12_most_freq_used_specialized_hardware_GPU',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs':'q12_most_freq_used_specialized_hardware_TPU',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None':'q12_most_freq_used_specialized_hardware_None',
 'Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':'q12_most_freq_used_specialized_hardware_Other',
 'Approximately how many times have you used a TPU (tensor processing unit)?':'q13_TPU_times_used',
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib ': "q14_most_freq_used_data_viz_library_Matplotlib",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn ': "q14_most_freq_used_data_viz_library_Seaborn",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express ': "q14_most_freq_used_data_viz_library_Plotly",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 ':"q14_most_freq_used_data_viz_library_Ggplot_ggplot2",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny ':"q14_most_freq_used_data_viz_library_Shiny",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js ':"q14_most_freq_used_data_viz_library_D3 js",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair ':"q14_most_freq_used_data_viz_library_Altair",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh ':"q14_most_freq_used_data_viz_library_Bokeh",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib ':"q14_most_freq_used_data_viz_library_Geoplotlib",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium ':"q14_most_freq_used_data_viz_library_Leaflet_Folium",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None':"q14_most_freq_used_data_viz_library_None",
 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':"q14_most_freq_used_data_viz_library_Other",
 'For how many years have you used machine learning methods?': 'q15_ml_methods_years_experience',
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn ':"q16_most_freq_used_ml_framework_Scikitlearn",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow ':"q16_most_freq_used_ml_framework_TensorFlow",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras ':"q16_most_freq_used_ml_framework_Keras",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch ':"q16_most_freq_used_ml_framework_PyTorch",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai ':"q16_most_freq_used_ml_framework_Fast.ai",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  MXNet ':"q16_most_freq_used_ml_framework_MXNet",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost ':"q16_most_freq_used_ml_framework_Xgboost",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM ':"q16_most_freq_used_ml_framework_LightGBM",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  CatBoost ':"q16_most_freq_used_ml_framework_CatBoost",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Prophet ':"q16_most_freq_used_ml_framework_Prophet",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  H2O 3 ':"q16_most_freq_used_ml_framework_H2O 3",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret ':"q16_most_freq_used_ml_framework_Caret",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Tidymodels ':"q16_most_freq_used_ml_framework_Tidymodels",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  JAX ':"q16_most_freq_used_ml_framework_JAX",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - None':"q16_most_freq_used_ml_framework_None",
 'Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice - Other':"q16_most_freq_used_ml_framework_Other",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression':"q17_most_freq_used_ml_algorithm_Linear_Logistic_Regression",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests':"q17_most_freq_used_ml_algorithm_DecisionTrees_RandomForests",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)':"q17_most_freq_used_ml_algorithm_GradientBoostingMachines",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches':"q17_most_freq_used_ml_algorithm_BayesianApproaches",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches':"q17_most_freq_used_ml_algorithm_EvolutionaryApproaches",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)':"q17_most_freq_used_ml_algorithm_DenseNeuralNetworks",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks':"q17_most_freq_used_ml_algorithm_ConvolutionalNeuralNetworks",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks':"q17_most_freq_used_ml_algorithm_GenerativeAdversarialNetworks",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks':"q17_most_freq_used_ml_algorithm_RecurrentNeuralNetworks",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-3, etc)':"q17_most_freq_used_ml_algorithm_TransformerNetworks",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - None':"q17_most_freq_used_ml_algorithm_None",
 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Other':"q17_most_freq_used_ml_algorithm_Other",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - General purpose image/video tools (PIL, cv2, skimage, etc)':"q18_most_freq_used_computervision_method_Generalpurposeimage_video",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image segmentation methods (U-Net, Mask R-CNN, etc)':"q18_most_freq_used_computervision_method_Imagesegmentation",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Object detection methods (YOLOv3, RetinaNet, etc)':"q18_most_freq_used_computervision_method_Objectdetection",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)':"q18_most_freq_used_computervision_method_Imageclassification",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Generative Networks (GAN, VAE, etc)':"q18_most_freq_used_computervision_method_GenerativeNetworks",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None':"q18_most_freq_used_computervision_method_None",
 'Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':"q18_most_freq_used_computervision_method_Other",
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Word embeddings/vectors (GLoVe, fastText, word2vec)':'q19_most_freq_used_NLP_Word_embeddings_vectors',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Encoder-decorder models (seq2seq, vanilla transformers)':'q19_most_freq_used_NLP_Encoder_decorder_models',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Contextualized embeddings (ELMo, CoVe)':'q19_most_freq_used_NLP_Contextualized_embeddings',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-3, BERT, XLnet, etc)':'q19_most_freq_used_NLP_Transformer_language_models',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - None':'q19_most_freq_used_NLP_None',
 'Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':'q19_most_freq_used_NLP_Other',
 'What is the size of the company where you are employed?':'q20_size_company',
 'Approximately how many individuals are responsible for data science workloads at your place of business?':'q21_num_data_science_employees',
 'Does your current employer incorporate machine learning methods into their business?':'q22_machineLearning_company_business',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions':'q23_tasks_work_analyzeData',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data':'q23_tasks_work_buildingDataInfrastructure',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas':'q23_tasks_work_MLPrototypes',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows':'q23_tasks_work_ML_productImprovement',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models':'q23_tasks_work_improve_ML_Models',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning':'q23_tasks_work_research_on_ML',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work':'q23_tasks_work_None',
 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other':'q23_tasks_work_Other',
 'What is your current yearly compensation (approximate $USD)?':'q24_yearly_compensation_$USD',
 'Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?':'q25_budget_spent_ml_cloudComputing_$USD',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) ':'q26_most_freq_used_cloudComputing_pltfrm_AWS',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure ':'q26_most_freq_used_cloudComputing_pltfrm_MicrosoftAzure',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) ':'q26_most_freq_used_cloudComputing_pltfrm_GCP',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud / Red Hat ':'q26_most_freq_used_cloudComputing_pltfrm_IBMCloud_RedHat',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_OracleCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_SAPCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_SalesforceCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_VMwareCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_AlibabaCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Tencent Cloud ':'q26_most_freq_used_cloudComputing_pltfrm_TencentCloud',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - None':'q26_most_freq_used_cloudComputing_pltfrm_None',
 'Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other':'q26_most_freq_used_cloudComputing_pltfrm_Other',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon EC2 ':'q27_most_freq_used_cloudComputing_prod_AmazonEC2',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  AWS Lambda ':'q27_most_freq_used_cloudComputing_prod_AWSLambda',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Elastic Container Service ':'q27_most_freq_used_cloudComputing_prod_AmazonElasticContainerService',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cloud Services ':'q27_most_freq_used_cloudComputing_prod_AzureCloudServices',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure Container Instances ':'q27_most_freq_used_cloudComputing_prod_MicrosoftAzureContainerInstances',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Functions ':'q27_most_freq_used_cloudComputing_prod_AzureFunctions',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ':'q27_most_freq_used_cloudComputing_prod_GoogleCloudComputeEngine',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Functions ':'q27_most_freq_used_cloudComputing_prod_GoogleCloudFunctions',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Run ':'q27_most_freq_used_cloudComputing_prod_GoogleCloudRun',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud App Engine ':'q27_most_freq_used_cloudComputing_prod_GoogleCloudAppEngine',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - No / None':'q27_most_freq_used_cloudComputing_prod_None',
 'Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - Other':'q27_most_freq_used_cloudComputing_prod_Other',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon SageMaker ':'q28_most_freq_used_ml_prod_AmazonSageMaker',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Forecast ':'q28_most_freq_used_ml_prod_AmazonForecast',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Rekognition ':'q28_most_freq_used_ml_prod_AmazonRekognition',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio ':'q28_most_freq_used_ml_prod_AzureMachineLearningStudio',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cognitive Services ':'q28_most_freq_used_ml_prod_AzureCognitiveServices',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud AI Platform / Google Cloud ML Engine':'q28_most_freq_used_ml_prod_GoogleCloudAI',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Video AI ':'q28_most_freq_used_ml_prod_GoogleVideoAI',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Natural Language ':'q28_most_freq_used_ml_prod_GoogleCloudNaturalLanguage',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Vision AI ':'q28_most_freq_used_ml_prod_GoogleCloudVisionAI',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - No / None':'q28_most_freq_used_ml_prod_None',
 'Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - Other':'q28_most_freq_used_ml_prod_Other',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL ':'q29_most_freq_used_bigData_prod_MySQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL ':'q29_most_freq_used_bigData_prod_PostgresSQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite ':'q29_most_freq_used_bigData_prod_SQLite',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database ':'q29_most_freq_used_bigData_prod_OracleDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MongoDB ':'q29_most_freq_used_bigData_prod_MongoDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Snowflake ':'q29_most_freq_used_bigData_prod_Snowlake',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - IBM Db2 ':'q29_most_freq_used_bigData_prod_IBMDb2',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server ':'q29_most_freq_used_bigData_prod_MicrosoftSQLServer',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access ':'q29_most_freq_used_bigData_prod_MicrosoftAccess',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage ':'q29_most_freq_used_bigData_prod_MicrosoftAzureDataLakeStorage',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Redshift ':'q29_most_freq_used_bigData_prod_AmazonRedshift',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon Athena ':'q29_most_freq_used_bigData_prod_AmazonAthena',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon DynamoDB ':'q29_most_freq_used_bigData_prod_AmazonDynamoDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud BigQuery ':'q29_most_freq_used_bigData_prod_GoogleCloudBigQuery',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL ':'q29_most_freq_used_bigData_prod_GoogleCloudSQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud Firestore ':'q29_most_freq_used_bigData_prod_GoogleCloudFirestore',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - None':'q29_most_freq_used_bigData_prod_None',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Other':'q29_most_freq_used_bigData_prod_Other',
 'Which of the following big data products (relational database, data warehouse, data lake, or similar) do you use most often? - Selected Choice':'q30_most_often_used_bigData_prod',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon QuickSight':'q31_most_freq_used_BITool_AmazonQuickSight',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Power BI':'q31_most_freq_used_BITool_MicrosoftPowerBI',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Google Data Studio':'q31_most_freq_used_BITool_GoogleDataStudio',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Looker':'q31_most_freq_used_BITool_Looker',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Tableau':'q31_most_freq_used_BITool_Tableau',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Salesforce':'q31_most_freq_used_BITool_Salesforce',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Einstein Analytics':'q31_most_freq_used_BITool_EinsteinAnalytics',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Qlik':'q31_most_freq_used_BITool_Qlik',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Domo':'q31_most_freq_used_BITool_Domo',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - TIBCO Spotfire':'q31_most_freq_used_BITool_TIBCOSpotfire',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Alteryx ':'q31_most_freq_used_BITool_Alteryx',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Sisense ':'q31_most_freq_used_BITool_Sisense',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - SAP Analytics Cloud ':'q31_most_freq_used_BITool_SAPAnalyticsCloud',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - None':'q31_most_freq_used_BITool_None',
 'Which of the following business intelligence tools do you use on a regular basis? (Select all that apply) - Selected Choice - Other':'q31_most_freq_used_BITool_Other',
 'Which of the following business intelligence tools do you use most often? - Selected Choice':'q32_most_often_used_BITool',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)':'q33_most_frew_used_AutoML_AutomatedDataAugmentation',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)':'q33_most_frew_used_AutoML_AutomatedFeatureEngineering',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)':'q33_most_frew_used_AutoML_AutomatedModelSelection',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)':'q33_most_frew_used_AutoML_AutomatedModelArchitectureSearches',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)':'q33_most_frew_used_AutoML_AutomatedHyperparameterTuning',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)':'q33_most_frew_used_AutoML_MLPipelines',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - No / None':'q33_most_frew_used_AutoML_None',
 'Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Other':'q33_most_frew_used_AutoML_Other',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud AutoML ':'q34_most_frew_used_AutoML_GoogleCloudAutoML',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ':'q34_most_frew_used_AutoML_H20DriverlessAI',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ':'q34_most_frew_used_AutoML_DatabricksAutoML',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ':'q34_most_frew_used_AutoML_DataRobotAutoML',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Tpot ':'q34_most_frew_used_AutoML_Tpot',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Keras ':'q34_most_frew_used_AutoML_AutoKeras',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto-Sklearn ':'q34_most_frew_used_AutoML_AutoSklearn',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Auto_ml ':'q34_most_frew_used_AutoML_Auto_ml',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Xcessiv ':'q34_most_frew_used_AutoML_Xcessiv',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -   MLbox ':'q34_most_frew_used_AutoML_MLbox',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - No / None':'q34_most_frew_used_AutoML_None',
 'Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other':'q34_most_frew_used_AutoML_Other',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Neptune.ai ':'q35_ml_experiments_tools_Neptune.ai',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Weights & Biases ':'q35_ml_experiments_tools_WeightsandBiases',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Comet.ml ':'q35_ml_experiments_tools_Comet.ml',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Sacred + Omniboard ':'q35_ml_experiments_tools_Sacred_Omniboard',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  TensorBoard ':'q35_ml_experiments_tools_TensorBoard',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Guild.ai ':'q35_ml_experiments_tools_Guild.ai',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Polyaxon ':'q35_ml_experiments_tools_Polyaxon',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Trains ':'q35_ml_experiments_tools_Trains',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice -  Domino Model Monitor ':'q35_ml_experiments_tools_DominoModelMonitor',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - No / None':'q35_ml_experiments_tools_None',
 'Do you use any tools to help manage machine learning experiments? (Select all that apply) - Selected Choice - Other':'q35_ml_experiments_tools_Other',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Plotly Dash ':'q36_deplpoyment_pltfrm_PlotlyDash',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Streamlit ':'q36_deplpoyment_pltfrm_Streamlit',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  NBViewer ':'q36_deplpoyment_pltfrm_NBViewer',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  GitHub ':'q36_deplpoyment_pltfrm_GitHub',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Personal blog ':'q36_deplpoyment_pltfrm_PersonalBlog',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Kaggle ':'q36_deplpoyment_pltfrm_Kaggle',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Colab ':'q36_deplpoyment_pltfrm_Colab',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice -  Shiny ':'q36_deplpoyment_pltfrm_Shiny',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - I do not share my work publicly':'q36_deplpoyment_pltfrm_notsharing',
 'Where do you publicly share or deploy your data analysis or machine learning applications? (Select all that apply) - Selected Choice - Other':'q36_deplpoyment_pltfrm_Other',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera':'q37_dataScience_courses_pltfrm_Coursera',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX':'q37_dataScience_courses_pltfrm_edX',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Learn Courses':'q37_dataScience_courses_pltfrm_KaggleLearnCourses',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp':'q37_dataScience_courses_pltfrm_DataCamp',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai':'q37_dataScience_courses_pltfrm_Fast.ai',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity':'q37_dataScience_courses_pltfrm_Udacity',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy':'q37_dataScience_courses_pltfrm_Udemy',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning':'q37_dataScience_courses_pltfrm_LinkedInLearning',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Cloud-certification programs (direct from AWS, Azure, GCP, or similar)':'q37_dataScience_courses_pltfrm_CloudCertifProgr',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)':'q37_dataScience_courses_pltfrm_University',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - None':'q37_dataScience_courses_pltfrm_None',
 'On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Other':'q37_dataScience_courses_pltfrm_Other',
 'What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice':'q38_primary_dataAnalysis_tool',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)':'q39_fav_media_sources_Twitter',
 "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Email newsletters (Data Elixir, O'Reilly Data & AI, etc)":'q39_fav_media_sources_Newsletter',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, etc)':'q39_fav_media_sources_Reddit',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (notebooks, forums, etc)':'q39_fav_media_sources_Kaggle',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, Coursera forums, etc)':'q39_fav_media_sources_CourseForums',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Kaggle YouTube, Cloud AI Adventures, etc)':'q39_fav_media_sources_YouTube',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)':'q39_fav_media_sources_Podcasts',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Analytics Vidhya, etc)':'q39_fav_media_sources_Blogs',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (peer-reviewed journals, conference proceedings, etc)':'q39_fav_media_sources_Journals',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)':'q39_fav_media_sources_Slack',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - None':'q39_fav_media_sources_None',
 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Other':'q39_fav_media_sources_Other',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Amazon Web Services (AWS) ':'q40_study_cloudComputing_pltfrm_nxt2year_AWS',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Microsoft Azure ':'q40_study_cloudComputing_pltfrm_nxt2year_Azure',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Google Cloud Platform (GCP) ':'q40_study_cloudComputing_pltfrm_nxt2year_GCP',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  IBM Cloud / Red Hat ':'q40_study_cloudComputing_pltfrm_nxt2year_IBMCloud_RedHat',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Oracle Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_OracleCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  SAP Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_SAPCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  VMware Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_VMwareCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Salesforce Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_SalesforceCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Alibaba Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_AlibabaCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice -  Tencent Cloud ':'q40_study_cloudComputing_pltfrm_nxt2year_TencentCloud',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice - None':'q40_study_cloudComputing_pltfrm_nxt2year_None',
 'Which of the following cloud computing platforms do you hope to become more familiar with in the next 2 years? - Selected Choice - Other':'q40_study_cloudComputing_pltfrm_nxt2year_Other',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Amazon EC2 ':'q41_study_cloudComputing_prod_nxt2year_AmazonEC2',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  AWS Lambda ':'q41_study_cloudComputing_prod_nxt2year_AWSLambda',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Amazon Elastic Container Service ':'q41_study_cloudComputing_prod_nxt2year_AmazonElasticContainerService',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Azure Cloud Services ':'q41_study_cloudComputing_prod_nxt2year_AzureCloudServices',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Microsoft Azure Container Instances ':'q41_study_cloudComputing_prod_nxt2year_MicrosoftAzureContainerInstances',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Azure Functions ':'q41_study_cloudComputing_prod_nxt2year_AzureFunctions',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ':'q41_study_cloudComputing_prod_nxt2year_GoogleCloudComputeEngine',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Functions ':'q41_study_cloudComputing_prod_nxt2year_GoogleCloudFunctions',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud Run ':'q41_study_cloudComputing_prod_nxt2year_GoogleCloudRun',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice -  Google Cloud App Engine ':'q41_study_cloudComputing_prod_nxt2year_GoogleCloudAppEngine',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice - None':'q41_study_cloudComputing_prod_nxt2year_None',
 'In the next 2 years, do you hope to become more familiar with any of these specific cloud computing products? (Select all that apply) - Selected Choice - Other':'q41_study_cloudComputing_prod_nxt2year_Other',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon SageMaker ':'q42_study_ml_prod_nxt2year_AmazonSageMaker',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon Forecast ':'q42_study_ml_prod_nxt2year_AmazonForecast',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Amazon Rekognition ':'q42_study_ml_prod_nxt2year_AmazonRekognition',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio ':'q42_study_ml_prod_nxt2year_AzureMachineLearningStudio',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Azure Cognitive Services ':'q42_study_ml_prod_nxt2year_AzureCognitiveServices',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud AI Platform / Google Cloud ML Engine':'q42_study_ml_prod_nxt2year_GoogleCloudAI_MLEngine',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Video AI ':'q42_study_ml_prod_nxt2year_GoogleCloudVideoAI',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Natural Language ':'q42_study_ml_prod_nxt2year_GoogleCloudNaturalLanguage',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice -  Google Cloud Vision AI ':'q42_study_ml_prod_nxt2year_GoogleCloudVisionAI',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice - None':'q42_study_ml_prod_nxt2year_None',
 'In the next 2 years, do you hope to become more familiar with any of these specific machine learning products? (Select all that apply) - Selected Choice - Other':'q42_study_ml_prod_nxt2year_Other',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - MySQL ':'q43_study_bigData_prod_nxt2year_MySQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - PostgresSQL ':'q43_study_bigData_prod_nxt2year_PostgresSQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - SQLite ':'q43_study_bigData_prod_nxt2year_SQLite',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Oracle Database ':'q43_study_bigData_prod_nxt2year_OracleDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - MongoDB ':'q43_study_bigData_prod_nxt2year_MongoDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Snowflake ':'q43_study_bigData_prod_nxt2year_Snowflake',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - IBM Db2 ':'q43_study_bigData_prod_nxt2year_IBMDb2',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft SQL Server ':'q43_study_bigData_prod_nxt2year_MicrosoftSQLServer',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Access ':'q43_study_bigData_prod_nxt2year_MicrosoftAccess',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage ':'q43_study_bigData_prod_nxt2year_MicrosoftAzureDataLakeStorage',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon Redshift ':'q43_study_bigData_prod_nxt2year_AmazonRedshift',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon Athena ':'q43_study_bigData_prod_nxt2year_AmazonAthena',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon DynamoDB ':'q43_study_bigData_prod_nxt2year_AmazonDynamoDB',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud BigQuery ':'q43_study_bigData_prod_nxt2year_GoogleCloudBigQuery',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud SQL ':'q43_study_bigData_prod_nxt2year_GoogleCloudSQL',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Cloud Firestore ':'q43_study_bigData_prod_nxt2year_GoogleCloudFirestore',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - None':'q43_study_bigData_prod_nxt2year_None',
 'Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Other':'q43_study_bigData_prod_nxt2year_Other',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Microsoft Power BI':'q44_study_BI_tool_nxt2year_MicrosoftPowerBI',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Amazon QuickSight':'q44_study_BI_tool_nxt2year_Amazon QuickSight',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Google Data Studio':'q44_study_BI_tool_nxt2year_GoogleDataStudio',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Looker':'q44_study_BI_tool_nxt2year_Looker',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Tableau':'q44_study_BI_tool_nxt2year_Tableau',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Salesforce':'q44_study_BI_tool_nxt2year_Salesforce',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Einstein Analytics':'q44_study_BI_tool_nxt2year_EinsteinAnalytics',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Qlik':'q44_study_BI_tool_nxt2year_Qlik',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Domo':'q44_study_BI_tool_nxt2year_Domo',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - TIBCO Spotfire':'q44_study_BI_tool_nxt2year_TIBCOSpotfire',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Alteryx ':'q44_study_BI_tool_nxt2year_Alteryx',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Sisense ':'q44_study_BI_tool_nxt2year_Sisense',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - SAP Analytics Cloud ':'q44_study_BI_tool_nxt2year_SAPAnalyticsCloud',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - None':'q44_study_BI_tool_nxt2year_None',
 'Which of the following business intelligence tools do you hope to become more familiar with in the next 2 years? (Select all that apply) - Selected Choice - Other':'q44_study_BI_tool_nxt2year_Other',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)':'q45_study_AutoML_tool_nxt2year_AutomatedDataAugmentation',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)':'q45_study_AutoML_tool_nxt2year_AutomatedFeatureEngineering',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)':'q45_study_AutoML_tool_nxt2year_AutomatedModelSelection',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)':'q45_study_AutoML_tool_nxt2year_AutomatedModelArchitecture',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)':'q45_study_AutoML_tool_nxt2year_AutomatedHyperparameterTuning',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google Cloud AutoML, H20 Driverless AI)':'q45_study_AutoML_tool_nxt2year_MLPipelines',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - None':'q45_study_AutoML_tool_nxt2year_None',
 'Which categories of automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Other':'q45_study_AutoML_tool_nxt2year_Other',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  Google Cloud AutoML ':'q46_study_AutoML_tool_nxt2year_GoogleCloudAutoML',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ':'q46_study_AutoML_tool_nxt2year_H20DriverlessAIL',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  Databricks AutoML ':'q46_study_AutoML_tool_nxt2year_DatabricksAutoML',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -  DataRobot AutoML ':'q46_study_AutoML_tool_nxt2year_DataRobotAutoML',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Tpot ':'q46_study_AutoML_tool_nxt2year_Tpot',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto-Keras ':'q46_study_AutoML_tool_nxt2year_Auto-Keras',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto-Sklearn ':'q46_study_AutoML_tool_nxt2year_AutoSklearn',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Auto_ml ':'q46_study_AutoML_tool_nxt2year_Auto_ml',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   Xcessiv ':'q46_study_AutoML_tool_nxt2year_Xcessiv',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice -   MLbox ':'q46_study_AutoML_tool_nxt2year_MLbox',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - None':'q46_study_AutoML_tool_nxt2year_None',
 'Which specific automated machine learning tools (or partial AutoML tools) do you hope to become more familiar with in the next 2 years?  (Select all that apply) - Selected Choice - Other':'q46_study_AutoML_tool_nxt2year_Other',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Neptune.ai ':'q47_study_MLExperiments_tool_nxt2year_Neptune.ai',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Weights & Biases ':'q47_study_MLExperiments_tool_nxt2year_WeightsandBiases',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Comet.ml ':'q47_study_MLExperiments_tool_nxt2year_Comet.ml',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Sacred + Omniboard ':'q47_study_MLExperiments_tool_nxt2year_SacredOmniboard',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  TensorBoard ':'q47_study_MLExperiments_tool_nxt2year_TensorBoard',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Guild.ai ':'q47_study_MLExperiments_tool_nxt2year_Guild.ai',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Polyaxon ':'q47_study_MLExperiments_tool_nxt2year_Polyaxon',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Trains ':'q47_study_MLExperiments_tool_nxt2year_Trains',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice -  Domino Model Monitor ':'q47_study_MLExperiments_tool_nxt2year_DominoModelMonitor',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice - None':'q47_study_MLExperiments_tool_nxt2year_None',
 'In the next 2 years, do you hope to become more familiar with any of these tools for managing ML experiments? (Select all that apply) - Selected Choice - Other':'q47_study_MLExperiments_tool_nxt2year_Other',
}


# In[5]:


#Here we are inserting our indivually columns to our df
df.rename(columns=rename_columns,inplace=True)


# In[6]:


#Getting names of columns (copy in the format file)
col_names = df.columns.tolist()


# <h2>Number of columns and rows</h2>

# In[7]:


num_columns = df.shape[1]
num_rows = df.shape[0]

print("Our DataFrame has 20036 rows x 355 cols")


#We could also use df.info to get num of cols and rows


# <h2>Checking datatypes</h2>

# In[8]:


df.dtypes

#The data type of each column is an object (we can ignore the first col "Duration" for now, we will change the dtype to int later)


# <h3>Checking precise dtype of each column</h3>

# In[9]:


#The following function checks if the individual dtypes of all the 20036 columns rows are either str or NaN (float).


def dftype_checker(index_num_column):
    """
Args:
Index number of column

Return:
Number of columns which have the object dtype string and/or NaN.
The purpose behind this function is to check if there are any suspicious values.
  
        
    """
    counter = 0
    for i in range (num_rows):
        if type(df.iloc[i,index_num_column]) == str or type(df.iloc[i,index_num_column])== float:
            counter+=1
    return counter


#Except the first column "Time from Start to Finish (seconds)" all the other columns entries are either str or NAN.
#We are going to change the df type from the "Time from Start to Finish (seconds)" column later


# In[10]:


#Evoking dftype_checker function by checking col index 9
dftype_checker(9) #Add the col index number year and print result


# <h2>Missing values rate</h2>

# In[11]:


#Displays list which entails column name and associated percentage of NaN's
percent_missing = round(df.isnull().sum() * 100 / len(df),2)

#Displays the DataFrame which entails column name and associated percentage of NaN's as seperate columns in df
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

#Displays only the percentages (ready to copy output)
percent_missing_only_percentages = ((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2).tolist()

#As the index progresses, the number of NaN Values also increases. 
#The first 4 questions were filled by 100% of the participants. 
#In the last question, only 1.35% filled out the form.

missing_value_df


# <h2>Distribution of values (Balance)</h2>

# <h5>Helper function to convert counts to percentages for selected Choice questions</h5>

# In[12]:


def count_then_return_percent(dataframe,column_name):
    '''
      Args:
        dataframe name and col name
        example: count_then_return_percent(df,'q5_professional_role')

    Returns:
        returns value counts as percentages.
    
    '''
    
    counts = dataframe[column_name].value_counts(dropna=False)
    percentages = round(counts*100/(dataframe[column_name].count()),1)
    return percentages

count_then_return_percent(df,'q20_size_company')


# <h4>Lists of answer choices and dictionaries of value counts (for the multiple choice & multiple selection questions)</h4>

# In[13]:


q7_most_freq_used_prgmg_languages = {
    'Python' : (df[col_names[7]].count()),
    'R': (df[col_names[8]].count()),
    'SQL' : (df[col_names[9]].count()),
    'C' : (df[col_names[10]].count()),
    'C++' : (df[col_names[11]].count()),
    'Java' : (df[col_names[12]].count()),
    'Javascript' : (df[col_names[13]].count()),
    'Julia' : (df[col_names[14]].count()),
    'Swift' : (df[col_names[15]].count()),
    'Bash' : (df[col_names[16]].count()),
    'MATLAB' : (df[col_names[17]].count()),
    'None' : (df[col_names[18]].count()),
    'Other' : (df[col_names[19]].count()),}

q9_most_freq_used_IDE = {
    'JupyterLab' : (df[col_names[21]].count()),
    'RStudio': (df[col_names[22]].count()),
    'Visual Studio' : (df[col_names[23]].count()),
    'Visual Studio Code (VSCode)' : (df[col_names[24]].count()),
    'PyCharm' : (df[col_names[25]].count()),
    'Spyder' : (df[col_names[26]].count()),
    'Notepad++' : (df[col_names[27]].count()),
    'Sublime Text' : (df[col_names[28]].count()),
    'Vim, Emacs, or similar' : (df[col_names[29]].count()),
    'MATLAB' : (df[col_names[30]].count()),
    'None' : (df[col_names[31]].count()),
    'Other' : (df[col_names[32]].count()),
}


q10_most_freq_used_notebook_products = {
    'Kaggle Notebooks' : (df[col_names[33]].count()),
    'Colab Notebooks': (df[col_names[34]].count()),
    'Azure Notebooks' : (df[col_names[35]].count()),
    'Paperspace / Gradient' : (df[col_names[36]].count()),
    'Binder / JupyterHub' : (df[col_names[37]].count()),
    'Code Ocean' : (df[col_names[38]].count()),
    'IBM Watson Studio' : (df[col_names[39]].count()),
    'Amazon Sagemaker Studio' : (df[col_names[40]].count()),
    'Amazon EMR Notebooks' : (df[col_names[41]].count()),
    'Google Cloud AI Platform Notebooks' : (df[col_names[42]].count()),
    'Google Cloud Datalab Notebooks' : (df[col_names[43]].count()),
    'Databricks Collaborative Notebooks' : (df[col_names[44]].count()),
    'None' : (df[col_names[45]].count()),
    'Other' : (df[col_names[46]].count()),
}


q12_specialized_hardware = {
    'GPUs' : (df[col_names[48]].count()),
    'TPUs': (df[col_names[49]].count()),
    'None' : (df[col_names[50]].count()),
    'Other' : (df[col_names[51]].count()),
}


q14_data_viz_libraries = {
    'Matplotlib' : (df[col_names[53]].count()),
    'Seaborn': (df[col_names[54]].count()),
    'Plotly / Plotly Express' : (df[col_names[55]].count()),
    'Ggplot / ggplot2' : (df[col_names[56]].count()),
    'Shiny' : (df[col_names[57]].count()),
    'D3.js' : (df[col_names[58]].count()),
    'Altair' : (df[col_names[59]].count()),
    'Bokeh' : (df[col_names[60]].count()),
    'Geoplotlib' : (df[col_names[61]].count()),
    'Leaflet / Folium' : (df[col_names[62]].count()),
    'None' : (df[col_names[63]].count()),
    'Other' : (df[col_names[64]].count()),
}


q16_machine_learning_frameworks = {
    'Scikit-learn' : (df[col_names[66]].count()),
    'TensorFlow': (df[col_names[67]].count()),
    'Keras' : (df[col_names[68]].count()),
    'PyTorch' : (df[col_names[69]].count()),
    'Fast.ai' : (df[col_names[70]].count()),
    'MXNet' : (df[col_names[71]].count()),
    'Xgboost' : (df[col_names[72]].count()),
    'LightGBM' : (df[col_names[73]].count()),
    'CatBoost' : (df[col_names[74]].count()),
    'Prophet' : (df[col_names[75]].count()),
    'H20-3' : (df[col_names[76]].count()),
    'Caret' : (df[col_names[77]].count()),
    'Tidymodels' : (df[col_names[78]].count()),
    'JAX' : (df[col_names[79]].count()),
    'None' : (df[col_names[80]].count()),
    'Other' : (df[col_names[81]].count()),
}

q17_ml_algorithms = {
    'Linear or Logistic Regression' : (df[col_names[82]].count()),
    'Decision Trees or Random Forests': (df[col_names[83]].count()),
    'Gradient Boosting Machines (xgboost, lightgbm, etc)' : (df[col_names[84]].count()),
    'Bayesian Approaches' : (df[col_names[85]].count()),
    'Evolutionary Approaches' : (df[col_names[86]].count()),
    'Dense Neural Networks (MLPs, etc)' : (df[col_names[87]].count()),
    'Convolutional Neural Networks' : (df[col_names[88]].count()),
    'Generative Adversarial Networks' : (df[col_names[89]].count()),
    'Recurrent Neural Networks' : (df[col_names[90]].count()),
    'Transformer Networks (BERT, gpt-3, etc)' : (df[col_names[91]].count()),
    'None' : (df[col_names[92]].count()),
    'Other' : (df[col_names[93]].count()),
}

q18_computer_vision_methods = {
    'General purpose image/video tools (PIL, cv2, skimage, etc)' : (df[col_names[94]].count()),
    'Image segmentation methods (U-Net, Mask R-CNN, etc)': (df[col_names[95]].count()),
    'Object detection methods (YOLOv3, RetinaNet, etc)' : (df[col_names[96]].count()),
    'Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)' : (df[col_names[97]].count()),
    'Generative Networks (GAN, VAE, etc)' : (df[col_names[98]].count()),
    'None' : (df[col_names[99]].count()),
    'Other' : (df[col_names[100]].count()),
}

q19_nlp = {
    'Word embeddings/vectors (GLoVe, fastText, word2vec)' : (df[col_names[101]].count()),
    'Encoder-decoder models (seq2seq, vanilla transformers)': (df[col_names[102]].count()),
    'Contextualized embeddings (ELMo, CoVe)' : (df[col_names[103]].count()),
    'Transformer language models (GPT-3, BERT, XLnet, etc)' : (df[col_names[104]].count()),
    'None' : (df[col_names[105]].count()),
    'Other' : (df[col_names[106]].count()),
}

q23_work_activity = {
    'Analyze and understand data to influence product or business decisions' : (df[col_names[110]].count()),
    'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data': (df[col_names[111]].count()),
    'Build prototypes to explore applying machine learning to new areas' : (df[col_names[112]].count()),
    'Build and/or run a machine learning service that operationally improves my product or workflows' : (df[col_names[113]].count()),
    'Experimentation and iteration to improve existing ML models' : (df[col_names[114]].count()),
    'Do research that advances the state of the art of machine learning' : (df[col_names[115]].count()),
    'None of these activities are an important part of my role at work' : (df[col_names[116]].count()),
    'Other' : (df[col_names[117]].count()),
}



q26a_most_freq_used_cloud_computing_platforms = {
    'Amazon Web Services (AWS)' : (df[col_names[120]].count()),
    'Microsoft Azure': (df[col_names[121]].count()),
    'Google Cloud Platform (GCP)' : (df[col_names[122]].count()),
    'IBM Cloud / Red Hat' :(df[col_names[123]].count()),
    'Oracle Cloud' : (df[col_names[124]].count()),
    'SAP Cloud' : (df[col_names[125]].count()),
    'Salesforce Cloud' : (df[col_names[126]].count()),
    'VMware Cloud' : (df[col_names[127]].count()),
    'Alibaba Cloud' : (df[col_names[128]].count()),
    'Tencent Cloud' : (df[col_names[129]].count()),
    'None' : (df[col_names[130]].count()),
    'Other' : (df[col_names[131]].count()),
}

q27a_cloud_computing_products = {
    'Amazon EC2' : (df[col_names[132]].count()),
    'AWS Lambda': (df[col_names[133]].count()),
    'Amazon Elastic Container Service' : (df[col_names[134]].count()),
    'Azure Cloud Services' : (df[col_names[135]].count()),
    'Microsoft Azure Container Instances' : (df[col_names[136]].count()),
    'Azure Functions' : (df[col_names[137]].count()),
    'Google Cloud Compute Engine' : (df[col_names[138]].count()),
    'Google Cloud Functions' : (df[col_names[139]].count()),
    'Google Cloud Run' : (df[col_names[140]].count()),
    'Google Cloud App Engine' : (df[col_names[141]].count()),
    'No / None' : (df[col_names[142]].count()),
    'Other' : (df[col_names[143]].count()),
}

q28a_machine_learning_products = {
    'Amazon SageMaker' : (df[col_names[144]].count()),
    'Amazon Forecast': (df[col_names[145]].count()),
    'Amazon Rekognition' : (df[col_names[146]].count()),
    'Azure Machine Learning Studio' : (df[col_names[147]].count()),
    'Azure Cognitive Services' : (df[col_names[148]].count()),
    'Google Cloud AI Platform / Google Cloud ML Engine' : (df[col_names[149]].count()),
    'Google Cloud Video AI' : (df[col_names[150]].count()),
    'Google Cloud Natural Language' : (df[col_names[151]].count()),
    'Google Cloud Vision AI' : (df[col_names[152]].count()),
    'No / None' : (df[col_names[153]].count()),
    'Other' : (df[col_names[154]].count()),
}

q29a_big_data_products = {
    'MySQL' : (df[col_names[155]].count()),
    'PostgreSQL': (df[col_names[156]].count()),
    'SQLite' : (df[col_names[157]].count()),
    'Oracle Database' : (df[col_names[158]].count()),
    'MongoDB' : (df[col_names[159]].count()),
    'Snowflake' : (df[col_names[160]].count()),
    'IBM Db2' : (df[col_names[161]].count()),
    'Microsoft SQL Server' : (df[col_names[162]].count()),
    'Microsoft Access' : (df[col_names[163]].count()),
    'Microsoft Azure Data Lake Storage' : (df[col_names[164]].count()),
    'Amazon Redshift' : (df[col_names[165]].count()),
    'Amazon Athena' : (df[col_names[166]].count()),
    'Amazon DynamoDB' : (df[col_names[167]].count()),
    'Google Cloud BigQuery' : (df[col_names[168]].count()),
    'Google Cloud SQL' : (df[col_names[169]].count()),
    'Google Cloud Firestore' : (df[col_names[170]].count()),
    'None' : (df[col_names[171]].count()),
    'Other' : (df[col_names[172]].count()),
}

q31a_business_intelligence_tools = {
    'Amazon QuickSight' : (df[col_names[174]].count()),
    'Microsoft Power BI': (df[col_names[175]].count()),
    'Google Data Studio' : (df[col_names[176]].count()),
    'Looker' : (df[col_names[178]].count()),
    'Tableau' : (df[col_names[179]].count()),
    'Salesforce' : (df[col_names[180]].count()),
    'Einstein Analytics' :(df[col_names[181]].count()),
    'Qlik' : (df[col_names[182]].count()),
    'Domo' : (df[col_names[183]].count()),
    'TIBCO Spotfire' : (df[col_names[184]].count()),
    'Alteryx' : (df[col_names[185]].count()),
    'Sisense' : (df[col_names[186]].count()),
    'SAP Analytics Cloud' : (df[col_names[187]].count()),
    'None' : (df[col_names[188]].count()),
    'Other' : (df[col_names[189]].count()),
}

q33a_automated_machine_learning_tools = {
    'Automated data augmentation (e.g. imgaug, albumentations)' : (df[col_names[190]].count()),
    'Automated feature engineering/selection (e.g. tpot, boruta_py)': (df[col_names[191]].count()),
    'Automated model selection (e.g. auto-sklearn, xcessiv)' : (df[col_names[192]].count()),
    'Automated model architecture searches (e.g. darts, enas)' : (df[col_names[193]].count()),
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)' : (df[col_names[194]].count()),
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (df[col_names[195]].count()),
    'No / None' : (df[col_names[196]].count()),
    'Other' : (df[col_names[197]].count()),
}


q34a_automated_machine_learning_tools = {
    'Google Cloud AutoML' : (df[col_names[198]].count()),
    'H20 Driverless AI': (df[col_names[199]].count()),
    'Databricks AutoML' : (df[col_names[200]].count()),
    'DataRobot AutoML' : (df[col_names[201]].count()),
    'Tpot' : (df[col_names[202]].count()),
    'Auto-Keras' : (df[col_names[203]].count()),
    'Auto-Sklearn' : (df[col_names[204]].count()),
    'Auto_ml' : (df[col_names[205]].count()),
    'Xcessiv' : (df[col_names[206]].count()),
    'MLbox' : (df[col_names[207]].count()),
    'No / None' : (df[col_names[208]].count()),
    'Other' : (df[col_names[209]].count()),
}


q35a_machine_learning_experiments = {
    'Neptune.ai' : (df[col_names[210]].count()),
    'Weights & Biases': (df[col_names[211]].count()),
    'Comet.ml' : (df[col_names[212]].count()),
    'Sacred + Omniboard' : (df[col_names[213]].count()),
    'TensorBoard' : (df[col_names[214]].count()),
    'Guild.ai' : (df[col_names[215]].count()),
    'Polyaxon' : (df[col_names[216]].count()),
    'Trains' : (df[col_names[217]].count()),
    'Domino Model Monitor' : (df[col_names[218]].count()),
    'No / None' : (df[col_names[219]].count()),
    'Other' : (df[col_names[220]].count()),
}



q36_deployment_platforms = {
    'Plotly Dash' : (df[col_names[221]].count()),
    'Streamlit': (df[col_names[222]].count()),
    'NBViewer' : (df[col_names[223]].count()),
    'GitHub' : (df[col_names[224]].count()),
    'Personal Blog' : (df[col_names[225]].count()),
    'Kaggle' : (df[col_names[226]].count()),
    'Colab' : (df[col_names[227]].count()),
    'Shiny' : (df[col_names[228]].count()),
    'None / I do not share my work publicly' :(df[col_names[229]].count()),
    'Other' : (df[col_names[230]].count()),
}

q37_data_science_courses = {
    'Coursera' : (df[col_names[231]].count()),
    'EdX': (df[col_names[232]].count()),
    'Kaggle Learn Courses' : (df[col_names[233]].count()),
    'DataCamp' : (df[col_names[234]].count()),
    'Fast.ai' : (df[col_names[235]].count()),
    'Udacity' : (df[col_names[236]].count()),
    'Udemy' : (df[col_names[237]].count()),
    'LinkedIn Learning' : (df[col_names[238]].count()),
    'Cloud-certification programs' : (df[col_names[239]].count()),
    'University Courses' : (df[col_names[240]].count()),
    'None' : (df[col_names[241]].count()),
    'Other' : (df[col_names[242]].count()),
}


q39_media_sources = {
    'Twitter (data science influencers)' : (df[col_names[244]].count()),
    'Email newsletters (Data Elixir, OReilly Data & AI, etc)': (df[col_names[245]].count()),
    'Reddit (r/machinelearning, etc)' : (df[col_names[246]].count()),
    'Kaggle (notebooks, forums, etc)' : (df[col_names[247]].count()),
    'Course Forums (forums.fast.ai, Coursera forums, etc)' : (df[col_names[248]].count()),
    'YouTube (Kaggle YouTube, Cloud AI Adventures, etc)' : (df[col_names[249]].count()),
    'Podcasts (Chai Time Data Science, OReilly Data Show, etc)' : (df[col_names[250]].count()),
    'Blogs (Towards Data Science, Analytics Vidhya, etc)' : (df[col_names[251]].count()),
    'Journal Publications (peer-reviewed journals, conference proceedings, etc)' : (df[col_names[252]].count()),
    'Slack Communities (ods.ai, kagglenoobs, etc)' : (df[col_names[253]].count()),
    'None' : (df[col_names[254]].count()),
    'Other' : (df[col_names[255]].count()),
}


q40_most_freq_used_cloud_computing_platforms = {
    'Amazon Web Services (AWS)' : (df[col_names[256]].count()),
    'Microsoft Azure': (df[col_names[257]].count()),
    'Google Cloud Platform (GCP)' : (df[col_names[258]].count()),
    'IBM Cloud / Red Hat' :(df[col_names[259]].count()),
    'Oracle Cloud' : (df[col_names[260]].count()),
    'SAP Cloud' : (df[col_names[261]].count()),
    'Salesforce Cloud' : (df[col_names[262]].count()),
    'VMware Cloud' : (df[col_names[263]].count()),
    'Alibaba Cloud' : (df[col_names[264]].count()),
    'Tencent Cloud' : (df[col_names[265]].count()),
    'None' : (df[col_names[266]].count()),
    'Other' : (df[col_names[267]].count()),
}

q41_cloud_computing_products = {
    'Amazon EC2' : (df[col_names[268]].count()),
    'AWS Lambda': (df[col_names[269]].count()),
    'Amazon Elastic Container Service' : (df[col_names[270]].count()),
    'Azure Cloud Services' : (df[col_names[271]].count()),
    'Microsoft Azure Container Instances' : (df[col_names[272]].count()),
    'Azure Functions' : (df[col_names[273]].count()),
    'Google Cloud Compute Engine' : (df[col_names[274]].count()),
    'Google Cloud Functions' : (df[col_names[275]].count()),
    'Google Cloud Run' : (df[col_names[276]].count()),
    'Google Cloud App Engine' : (df[col_names[277]].count()),
    'No / None' : (df[col_names[278]].count()),
    'Other' : (df[col_names[279]].count()),
}

q42_machine_learning_products = {
    'Amazon SageMaker' : (df[col_names[280]].count()),
    'Amazon Forecast': (df[col_names[281]].count()),
    'Amazon Rekognition' : (df[col_names[282]].count()),
    'Azure Machine Learning Studio' : (df[col_names[283]].count()),
    'Azure Cognitive Services' : (df[col_names[284]].count()),
    'Google Cloud AI Platform / Google Cloud ML Engine' : (df[col_names[285]].count()),
    'Google Cloud Video AI' : (df[col_names[286]].count()),
    'Google Cloud Natural Language' : (df[col_names[287]].count()),
    'Google Cloud Vision AI' : (df[col_names[288]].count()),
    'No / None' : (df[col_names[289]].count()),
    'Other' : (df[col_names[290]].count()),
}

q43_big_data_products = {
    'MySQL' : (df[col_names[291]].count()),
    'PostgreSQL': (df[col_names[292]].count()),
    'SQLite' : (df[col_names[293]].count()),
    'Oracle Database' : (df[col_names[294]].count()),
    'MongoDB' : (df[col_names[295]].count()),
    'Snowflake' : (df[col_names[296]].count()),
    'IBM Db2' : (df[col_names[297]].count()),
    'Microsoft SQL Server' : (df[col_names[298]].count()),
    'Microsoft Access' : (df[col_names[299]].count()),
    'Microsoft Azure Data Lake Storage' : (df[col_names[300]].count()),
    'Amazon Redshift' : (df[col_names[301]].count()),
    'Amazon Athena' : (df[col_names[302]].count()),
    'Amazon DynamoDB' : (df[col_names[303]].count()),
    'Google Cloud BigQuery' : (df[col_names[304]].count()),
    'Google Cloud SQL' : (df[col_names[305]].count()),
    'Google Cloud Firestore' : (df[col_names[306]].count()),
    'None' : (df[col_names[307]].count()),
    'Other' : (df[col_names[308]].count()),
}

q44_business_intelligence_tools = {
    'Amazon QuickSight' : (df[col_names[309]].count()),
    'Microsoft Power BI': (df[col_names[310]].count()),
    'Google Data Studio' : (df[col_names[311]].count()),
    'Looker' : (df[col_names[312]].count()),
    'Tableau' : (df[col_names[313]].count()),
    'Salesforce' : (df[col_names[314]].count()),
    'Einstein Analytics' :(df[col_names[315]].count()),
    'Qlik' : (df[col_names[316]].count()),
    'Domo' : (df[col_names[317]].count()),
    'TIBCO Spotfire' : (df[col_names[318]].count()),
    'Alteryx' : (df[col_names[319]].count()),
    'Sisense' : (df[col_names[320]].count()),
    'SAP Analytics Cloud' : (df[col_names[321]].count()),
    'None' : (df[col_names[322]].count()),
    'Other' : (df[col_names[323]].count()),
}

q45_automated_machine_learning_tools = {
    'Automated data augmentation (e.g. imgaug, albumentations)' : (df[col_names[324]].count()),
    'Automated feature engineering/selection (e.g. tpot, boruta_py)': (df[col_names[325]].count()),
    'Automated model selection (e.g. auto-sklearn, xcessiv)' : (df[col_names[326]].count()),
    'Automated model architecture searches (e.g. darts, enas)' : (df[col_names[327]].count()),
    'Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)' : (df[col_names[328]].count()),
    'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' : (df[col_names[329]].count()),
    'No / None' : (df[col_names[330]].count()),
    'Other' : (df[col_names[331]].count()),
}

q46_automated_machine_learning_tools = {
    'Google Cloud AutoML' : (df[col_names[332]].count()),
    'H20 Driverless AI': (df[col_names[333]].count()),
    'Databricks AutoML' : (df[col_names[334]].count()),
    'DataRobot AutoML' : (df[col_names[335]].count()),
    'Tpot' : (df[col_names[336]].count()),
    'Auto-Keras' : (df[col_names[337]].count()),
    'Auto-Sklearn' : (df[col_names[338]].count()),
    'Auto_ml' : (df[col_names[339]].count()),
    'Xcessiv' : (df[col_names[340]].count()),
    'MLbox' : (df[col_names[341]].count()),
    'No / None' : (df[col_names[342]].count()),
    'Other' : (df[col_names[343]].count()),
}


q47_machine_learning_experiments = {
    'Neptune.ai' : (df[col_names[344]].count()),
    'Weights & Biases': (df[col_names[345]].count()),
    'Comet.ml' : (df[col_names[346]].count()),
    'Sacred + Omniboard' : (df[col_names[347]].count()),
    'TensorBoard' : (df[col_names[348]].count()),
    'Guild.ai' : (df[col_names[349]].count()),
    'Polyaxon' : (df[col_names[350]].count()),
    'Trains' : (df[col_names[351]].count()),
    'Domino Model Monitor' : (df[col_names[352]].count()),
    'No / None' : (df[col_names[353]].count()),
    'Other' : (df[col_names[354]].count()),
}



col_names[344:355]




# <h5>Helper function to convert counts to percentages for multiple choice/multiple selection questions</h5>

# In[14]:


def count_then_return_percent_for_multiple_column_questions(dataframe,list_of_columns_for_a_single_question,dictionary_of_counts_for_a_single_question):
    '''
    A helper function to convert counts to percentages for multiple column questions.
    
    Args:
    1.dataframe, 
    2.list of columns for a single_question (example: col_names[7:20]),
    3.dictionary_of_counts (indiviadual list for each multiple column questions)
    
    Example to evoke function: count_then_return_percent_for_multiple_column_questions(df,col_names[7:20],q7_most_freq_used_prgmg_languages)

    
    Returns:
    returns value counts as percentages.
    '''
    df = dataframe
    subset = list_of_columns_for_a_single_question
    df = df[subset]
    df = df.dropna(how='all')
    total_count = len(df) 
    dictionary = dictionary_of_counts_for_a_single_question
    for i in dictionary:
        dictionary[i] = round(float(dictionary[i]*100/total_count),1)
    return dictionary 

count_then_return_percent_for_multiple_column_questions(df,col_names[344:355],q47_machine_learning_experiments)


# <h5>Helper function to sort dict</h5>

# In[15]:


def sort_dictionary_by_percent(dataframe,list_of_columns_for_a_single_question,dictionary_of_counts_for_a_single_question): 
    ''' 
    A helper function that can be used to sort a dictionary.

    '''
    dictionary = count_then_return_percent_for_multiple_column_questions(dataframe,
                                                                list_of_columns_for_a_single_question,
                                                                dictionary_of_counts_for_a_single_question)
    dictionary = {v:k    for(k,v) in dictionary.items()}
    list_tuples = sorted(dictionary.items(), reverse=False) 
    dictionary = {v:k for (k,v) in list_tuples}   
    return dictionary


# <h1>Formatting and handling data</h1>

# <h2>Converting dtype of "duration" colunm from str to int</h2>

# In[16]:


df['duration_seconds'] = df['duration_seconds'].astype("int64")
df['duration_seconds'].dtype


# In[17]:


df.dtypes


# <h2>Grouping columns that can be assigned to a coherent multiple choice question</h2>

# In[18]:


q7 = [ 'q7_most_freq_used_prgmg_language_Python',
 'q7_most_freq_used_prgmg_language_R',
 'q7_most_freq_used_prgmg_language_SQL',
 'q7_most_freq_used_prgmg_language_C',
 'q7_most_freq_used_prgmg_language_C++',
 'q7_most_freq_used_prgmg_language_Java',
 'q7_most_freq_used_prgmg_language_Javascript',
 'q7_most_freq_used_prgmg_language_Julia',
 'q7_most_freq_used_prgmg_language_Swift',
 'q7_most_freq_used_prgmg_language_Bash',
 'q7_most_freq_used_prgmg_language_MATLAB',
 'q7_most_freq_used_prgmg_language_None',
 'q7_most_freq_used_prgmg_language_Other',]


q9 =['q9_most_freq_used_IDE_Jupyter',
 'q9_most_freq_used_IDE_RStudio',
 'q9_most_freq_used_IDE_Visual_Studio_Code',
 'q9_most_freq_used_IDE_Choice13',
 'q9_most_freq_used_IDE_PyCharm',
 'q9_most_freq_used_IDE_Spyder',
 'q9_most_freq_used_IDE_Notepad++',
 'q9_most_freq_used_IDE_SublimeText',
 'q9_most_freq_used_IDE_Vim_Emacs',
 'q9_most_freq_used_IDE_MATLAB',
 'q9_most_freq_used_IDE_None',
 'q9_most_freq_used_IDE_Other']

q10 = ['q10_most_freq_used_notebook_product_KaggleNotebooks',
 'q10_most_freq_used_notebook_product_ColabNotebooks',
 'q10_most_freq_used_notebook_product_AzureNotebooks',
 'q10_most_freq_used_notebook_product_Paperspace_Gradient',
 'q10_most_freq_used_notebook_product_Binder_JupyterHub',
 'q10_most_freq_used_notebook_product_CodeOcean',
 'q10_most_freq_used_notebook_product_IBMWatsonStudio',
 'q10_most_freq_used_notebook_product_AmazonSagemakerStudio',
 'q10_most_freq_used_notebook_product_AmazonEMRNotebooks',
 'q10_most_freq_used_notebook_product_GoogleCloudAIPlatformNotebooks',
 'q10_most_freq_used_notebook_product_GoogleCloudDatalabNotebooks',
 'q10_most_freq_used_notebook_product_DatabricksCollaborativeNotebooks',
 'q10_most_freq_used_notebook_product_None',
 'q10_most_freq_used_notebook_product_Other',]


q12= ['q12_most_freq_used_specialized_hardware_GPU',
 'q12_most_freq_used_specialized_hardware_TPU',
 'q12_most_freq_used_specialized_hardware_None',
 'q12_most_freq_used_specialized_hardware_Other',]


q14 = ['q14_most_freq_used_data_viz_library_Matplotlib',
 'q14_most_freq_used_data_viz_library_Seaborn',
 'q14_most_freq_used_data_viz_library_Plotly',
 'q14_most_freq_used_data_viz_library_Ggplot_ggplot2',
 'q14_most_freq_used_data_viz_library_Shiny',
 'q14_most_freq_used_data_viz_library_D3 js',
 'q14_most_freq_used_data_viz_library_Altair',
 'q14_most_freq_used_data_viz_library_Bokeh',
 'q14_most_freq_used_data_viz_library_Geoplotlib',
 'q14_most_freq_used_data_viz_library_Leaflet_Folium',
 'q14_most_freq_used_data_viz_library_None',
 'q14_most_freq_used_data_viz_library_Other']


q16 = ['q16_most_freq_used_ml_framework_Scikitlearn',
 'q16_most_freq_used_ml_framework_TensorFlow',
 'q16_most_freq_used_ml_framework_Keras',
 'q16_most_freq_used_ml_framework_PyTorch',
 'q16_most_freq_used_ml_framework_Fast.ai',
 'q16_most_freq_used_ml_framework_MXNet',
 'q16_most_freq_used_ml_framework_Xgboost',
 'q16_most_freq_used_ml_framework_LightGBM',
 'q16_most_freq_used_ml_framework_CatBoost',
 'q16_most_freq_used_ml_framework_Prophet',
 'q16_most_freq_used_ml_framework_H2O 3',
 'q16_most_freq_used_ml_framework_Caret',
 'q16_most_freq_used_ml_framework_Tidymodels',
 'q16_most_freq_used_ml_framework_JAX',
 'q16_most_freq_used_ml_framework_None',
 'q16_most_freq_used_ml_framework_Other',]


q17 = ['q17_most_freq_used_ml_algorithm_Linear_Logistic_Regression',
 'q17_most_freq_used_ml_algorithm_DecisionTrees_RandomForests',
 'q17_most_freq_used_ml_algorithm_GradientBoostingMachines',
 'q17_most_freq_used_ml_algorithm_BayesianApproaches',
 'q17_most_freq_used_ml_algorithm_EvolutionaryApproaches',
 'q17_most_freq_used_ml_algorithm_DenseNeuralNetworks',
 'q17_most_freq_used_ml_algorithm_ConvolutionalNeuralNetworks',
 'q17_most_freq_used_ml_algorithm_GenerativeAdversarialNetworks',
 'q17_most_freq_used_ml_algorithm_RecurrentNeuralNetworks',
 'q17_most_freq_used_ml_algorithm_TransformerNetworks',
 'q17_most_freq_used_ml_algorithm_None',
 'q17_most_freq_used_ml_algorithm_Other',]

q18 = [ 'q18_most_freq_used_computervision_method_Generalpurposeimage_video',
 'q18_most_freq_used_computervision_method_Imagesegmentation',
 'q18_most_freq_used_computervision_method_Objectdetection',
 'q18_most_freq_used_computervision_method_Imageclassification',
 'q18_most_freq_used_computervision_method_GenerativeNetworks',
 'q18_most_freq_used_computervision_method_None',
 'q18_most_freq_used_computervision_method_Other',]

q19 = ['q19_most_freq_used_NLP_Word_embeddings_vectors',
 'q19_most_freq_used_NLP_Encoder_decorder_models',
 'q19_most_freq_used_NLP_Contextualized_embeddings',
 'q19_most_freq_used_NLP_Transformer_language_models',
 'q19_most_freq_used_NLP_None',
 'q19_most_freq_used_NLP_Other',]


q23 = ['q23_tasks_work_analyzeData',
 'q23_tasks_work_buildingDataInfrastructure',
 'q23_tasks_work_MLPrototypes',
 'q23_tasks_work_ML_productImprovement',
 'q23_tasks_work_improve_ML_Models',
 'q23_tasks_work_research_on_ML',
 'q23_tasks_work_None',
 'q23_tasks_work_Other',]

q26 = ['q26_most_freq_used_cloudComputing_pltfrm_AWS',
 'q26_most_freq_used_cloudComputing_pltfrm_MicrosoftAzure',
 'q26_most_freq_used_cloudComputing_pltfrm_GCP',
 'q26_most_freq_used_cloudComputing_pltfrm_IBMCloud_RedHat',
 'q26_most_freq_used_cloudComputing_pltfrm_OracleCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_SAPCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_SalesforceCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_VMwareCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_AlibabaCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_TencentCloud',
 'q26_most_freq_used_cloudComputing_pltfrm_None',
 'q26_most_freq_used_cloudComputing_pltfrm_Other',]

q27 = ['q27_most_freq_used_cloudComputing_prod_AmazonEC2',
 'q27_most_freq_used_cloudComputing_prod_AWSLambda',
 'q27_most_freq_used_cloudComputing_prod_AmazonElasticContainerService',
 'q27_most_freq_used_cloudComputing_prod_AzureCloudServices',
 'q27_most_freq_used_cloudComputing_prod_MicrosoftAzureContainerInstances',
 'q27_most_freq_used_cloudComputing_prod_AzureFunctions',
 'q27_most_freq_used_cloudComputing_prod_GoogleCloudComputeEngine',
 'q27_most_freq_used_cloudComputing_prod_GoogleCloudFunctions',
 'q27_most_freq_used_cloudComputing_prod_GoogleCloudRun',
 'q27_most_freq_used_cloudComputing_prod_GoogleCloudAppEngine',
 'q27_most_freq_used_cloudComputing_prod_None',
 'q27_most_freq_used_cloudComputing_prod_Other',]


q28 = ['q28_most_freq_used_ml_prod_AmazonSageMaker',
 'q28_most_freq_used_ml_prod_AmazonForecast',
 'q28_most_freq_used_ml_prod_AmazonRekognition',
 'q28_most_freq_used_ml_prod_AzureMachineLearningStudio',
 'q28_most_freq_used_ml_prod_AzureCognitiveServices',
 'q28_most_freq_used_ml_prod_GoogleCloudAI',
 'q28_most_freq_used_ml_prod_GoogleVideoAI',
 'q28_most_freq_used_ml_prod_GoogleCloudNaturalLanguage',
 'q28_most_freq_used_ml_prod_GoogleCloudVisionAI',
 'q28_most_freq_used_ml_prod_None',
 'q28_most_freq_used_ml_prod_Other',]

q29 = ['q29_most_freq_used_bigData_prod_MySQL',
 'q29_most_freq_used_bigData_prod_PostgresSQL',
 'q29_most_freq_used_bigData_prod_SQLite',
 'q29_most_freq_used_bigData_prod_OracleDB',
 'q29_most_freq_used_bigData_prod_MongoDB',
 'q29_most_freq_used_bigData_prod_Snowlake',
 'q29_most_freq_used_bigData_prod_IBMDb2',
 'q29_most_freq_used_bigData_prod_MicrosoftSQLServer',
 'q29_most_freq_used_bigData_prod_MicrosoftAccess',
 'q29_most_freq_used_bigData_prod_MicrosoftAzureDataLakeStorage',
 'q29_most_freq_used_bigData_prod_AmazonRedshift',
 'q29_most_freq_used_bigData_prod_AmazonAthena',
 'q29_most_freq_used_bigData_prod_AmazonDynamoDB',
 'q29_most_freq_used_bigData_prod_GoogleCloudBigQuery',
 'q29_most_freq_used_bigData_prod_GoogleCloudSQL',
 'q29_most_freq_used_bigData_prod_GoogleCloudFirestore',
 'q29_most_freq_used_bigData_prod_None',
 'q29_most_freq_used_bigData_prod_Other',]

q31 = ['q31_most_freq_used_BITool_AmazonQuickSight',
 'q31_most_freq_used_BITool_MicrosoftPowerBI',
 'q31_most_freq_used_BITool_GoogleDataStudio',
 'q31_most_freq_used_BITool_Looker',
 'q31_most_freq_used_BITool_Tableau',
 'q31_most_freq_used_BITool_Salesforce',
 'q31_most_freq_used_BITool_EinsteinAnalytics',
 'q31_most_freq_used_BITool_Qlik',
 'q31_most_freq_used_BITool_Domo',
 'q31_most_freq_used_BITool_TIBCOSpotfire',
 'q31_most_freq_used_BITool_Alteryx',
 'q31_most_freq_used_BITool_Sisense',
 'q31_most_freq_used_BITool_SAPAnalyticsCloud',
 'q31_most_freq_used_BITool_None',
 'q31_most_freq_used_BITool_Other',]

q33 = ['q33_most_frew_used_AutoML_AutomatedDataAugmentation',
 'q33_most_frew_used_AutoML_AutomatedFeatureEngineering',
 'q33_most_frew_used_AutoML_AutomatedModelSelection',
 'q33_most_frew_used_AutoML_AutomatedModelArchitectureSearches',
 'q33_most_frew_used_AutoML_AutomatedHyperparameterTuning',
 'q33_most_frew_used_AutoML_MLPipelines',
 'q33_most_frew_used_AutoML_None',
 'q33_most_frew_used_AutoML_Other',]

q34 = ['q34_most_frew_used_AutoML_GoogleCloudAutoML',
 'q34_most_frew_used_AutoML_H20DriverlessAI',
 'q34_most_frew_used_AutoML_DatabricksAutoML',
 'q34_most_frew_used_AutoML_DataRobotAutoML',
 'q34_most_frew_used_AutoML_Tpot',
 'q34_most_frew_used_AutoML_AutoKeras',
 'q34_most_frew_used_AutoML_AutoSklearn',
 'q34_most_frew_used_AutoML_Auto_ml',
 'q34_most_frew_used_AutoML_Xcessiv',
 'q34_most_frew_used_AutoML_MLbox',
 'q34_most_frew_used_AutoML_None',
 'q34_most_frew_used_AutoML_Other',]


q35 = ['q35_ml_experiments_tools_Neptune.ai',
 'q35_ml_experiments_tools_WeightsandBiases',
 'q35_ml_experiments_tools_Comet.ml',
 'q35_ml_experiments_tools_Sacred_Omniboard',
 'q35_ml_experiments_tools_TensorBoard',
 'q35_ml_experiments_tools_Guild.ai',
 'q35_ml_experiments_tools_Polyaxon',
 'q35_ml_experiments_tools_Trains',
 'q35_ml_experiments_tools_DominoModelMonitor',
 'q35_ml_experiments_tools_None',
 'q35_ml_experiments_tools_Other',]

q36 = ['q36_deplpoyment_pltfrm_PlotlyDash',
 'q36_deplpoyment_pltfrm_Streamlit',
 'q36_deplpoyment_pltfrm_NBViewer',
 'q36_deplpoyment_pltfrm_GitHub',
 'q36_deplpoyment_pltfrm_PersonalBlog',
 'q36_deplpoyment_pltfrm_Kaggle',
 'q36_deplpoyment_pltfrm_Colab',
 'q36_deplpoyment_pltfrm_Shiny',
 'q36_deplpoyment_pltfrm_notsharing',
 'q36_deplpoyment_pltfrm_Other',]

q37 = ['q37_dataScience_courses_pltfrm_Coursera',
 'q37_dataScience_courses_pltfrm_edX',
 'q37_dataScience_courses_pltfrm_KaggleLearnCourses',
 'q37_dataScience_courses_pltfrm_DataCamp',
 'q37_dataScience_courses_pltfrm_Fast.ai',
 'q37_dataScience_courses_pltfrm_Udacity',
 'q37_dataScience_courses_pltfrm_Udemy',
 'q37_dataScience_courses_pltfrm_LinkedInLearning',
 'q37_dataScience_courses_pltfrm_CloudCertifProgr',
 'q37_dataScience_courses_pltfrm_University',
 'q37_dataScience_courses_pltfrm_None',
 'q37_dataScience_courses_pltfrm_Other',]

q39 = ['q39_fav_media_sources_Twitter',
 'q39_fav_media_sources_Newsletter',
 'q39_fav_media_sources_Reddit',
 'q39_fav_media_sources_Kaggle',
 'q39_fav_media_sources_CourseForums',
 'q39_fav_media_sources_YouTube',
 'q39_fav_media_sources_Podcasts',
 'q39_fav_media_sources_Blogs',
 'q39_fav_media_sources_Journals',
 'q39_fav_media_sources_Slack',
 'q39_fav_media_sources_None',
 'q39_fav_media_sources_Other',]

q40 = ['q40_study_cloudComputing_pltfrm_nxt2year_AWS',
 'q40_study_cloudComputing_pltfrm_nxt2year_Azure',
 'q40_study_cloudComputing_pltfrm_nxt2year_GCP',
 'q40_study_cloudComputing_pltfrm_nxt2year_IBMCloud_RedHat',
 'q40_study_cloudComputing_pltfrm_nxt2year_OracleCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_SAPCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_VMwareCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_SalesforceCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_AlibabaCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_TencentCloud',
 'q40_study_cloudComputing_pltfrm_nxt2year_None',
 'q40_study_cloudComputing_pltfrm_nxt2year_Other',]


q41 = ['q41_study_cloudComputing_prod_nxt2year_AmazonEC2',
 'q41_study_cloudComputing_prod_nxt2year_AWSLambda',
 'q41_study_cloudComputing_prod_nxt2year_AmazonElasticContainerService',
 'q41_study_cloudComputing_prod_nxt2year_AzureCloudServices',
 'q41_study_cloudComputing_prod_nxt2year_MicrosoftAzureContainerInstances',
 'q41_study_cloudComputing_prod_nxt2year_AzureFunctions',
 'q41_study_cloudComputing_prod_nxt2year_GoogleCloudComputeEngine',
 'q41_study_cloudComputing_prod_nxt2year_GoogleCloudFunctions',
 'q41_study_cloudComputing_prod_nxt2year_GoogleCloudRun',
 'q41_study_cloudComputing_prod_nxt2year_GoogleCloudAppEngine',
 'q41_study_cloudComputing_prod_nxt2year_None',
 'q41_study_cloudComputing_prod_nxt2year_Other',]

q42 = ['q42_study_ml_prod_nxt2year_AmazonSageMaker',
 'q42_study_ml_prod_nxt2year_AmazonForecast',
 'q42_study_ml_prod_nxt2year_AmazonRekognition',
 'q42_study_ml_prod_nxt2year_AzureMachineLearningStudio',
 'q42_study_ml_prod_nxt2year_AzureCognitiveServices',
 'q42_study_ml_prod_nxt2year_GoogleCloudAI_MLEngine',
 'q42_study_ml_prod_nxt2year_GoogleCloudVideoAI',
 'q42_study_ml_prod_nxt2year_GoogleCloudNaturalLanguage',
 'q42_study_ml_prod_nxt2year_GoogleCloudVisionAI',
 'q42_study_ml_prod_nxt2year_None',
 'q42_study_ml_prod_nxt2year_Other',]

q43 = [ 'q43_study_bigData_prod_nxt2year_MySQL',
 'q43_study_bigData_prod_nxt2year_PostgresSQL',
 'q43_study_bigData_prod_nxt2year_SQLite',
 'q43_study_bigData_prod_nxt2year_OracleDB',
 'q43_study_bigData_prod_nxt2year_MongoDB',
 'q43_study_bigData_prod_nxt2year_Snowflake',
 'q43_study_bigData_prod_nxt2year_IBMDb2',
 'q43_study_bigData_prod_nxt2year_MicrosoftSQLServer',
 'q43_study_bigData_prod_nxt2year_MicrosoftAccess',
 'q43_study_bigData_prod_nxt2year_MicrosoftAzureDataLakeStorage',
 'q43_study_bigData_prod_nxt2year_AmazonRedshift',
 'q43_study_bigData_prod_nxt2year_AmazonAthena',
 'q43_study_bigData_prod_nxt2year_AmazonDynamoDB',
 'q43_study_bigData_prod_nxt2year_GoogleCloudBigQuery',
 'q43_study_bigData_prod_nxt2year_GoogleCloudSQL',
 'q43_study_bigData_prod_nxt2year_GoogleCloudFirestore',
 'q43_study_bigData_prod_nxt2year_None',
 'q43_study_bigData_prod_nxt2year_Other',]


q44 = ['q44_study_BI_tool_nxt2year_MicrosoftPowerBI',
 'q44_study_BI_tool_nxt2year_Amazon QuickSight',
 'q44_study_BI_tool_nxt2year_GoogleDataStudio',
 'q44_study_BI_tool_nxt2year_Looker',
 'q44_study_BI_tool_nxt2year_Tableau',
 'q44_study_BI_tool_nxt2year_Salesforce',
 'q44_study_BI_tool_nxt2year_EinsteinAnalytics',
 'q44_study_BI_tool_nxt2year_Qlik',
 'q44_study_BI_tool_nxt2year_Domo',
 'q44_study_BI_tool_nxt2year_TIBCOSpotfire',
 'q44_study_BI_tool_nxt2year_Alteryx',
 'q44_study_BI_tool_nxt2year_Sisense',
 'q44_study_BI_tool_nxt2year_SAPAnalyticsCloud',
 'q44_study_BI_tool_nxt2year_None',
 'q44_study_BI_tool_nxt2year_Other',]

q45 = ['q45_study_AutoML_tool_nxt2year_AutomatedDataAugmentation',
 'q45_study_AutoML_tool_nxt2year_AutomatedFeatureEngineering',
 'q45_study_AutoML_tool_nxt2year_AutomatedModelSelection',
 'q45_study_AutoML_tool_nxt2year_AutomatedModelArchitecture',
 'q45_study_AutoML_tool_nxt2year_AutomatedHyperparameterTuning',
 'q45_study_AutoML_tool_nxt2year_MLPipelines',
 'q45_study_AutoML_tool_nxt2year_None',
 'q45_study_AutoML_tool_nxt2year_Other',]


q46 = [ 'q46_study_AutoML_tool_nxt2year_GoogleCloudAutoML',
 'q46_study_AutoML_tool_nxt2year_H20DriverlessAIL',
 'q46_study_AutoML_tool_nxt2year_DatabricksAutoML',
 'q46_study_AutoML_tool_nxt2year_DataRobotAutoML',
 'q46_study_AutoML_tool_nxt2year_Tpot',
 'q46_study_AutoML_tool_nxt2year_Auto-Keras',
 'q46_study_AutoML_tool_nxt2year_AutoSklearn',
 'q46_study_AutoML_tool_nxt2year_Auto_ml',
 'q46_study_AutoML_tool_nxt2year_Xcessiv',
 'q46_study_AutoML_tool_nxt2year_MLbox',
 'q46_study_AutoML_tool_nxt2year_None',
 'q46_study_AutoML_tool_nxt2year_Other',]

q47 = ['q47_study_MLExperiments_tool_nxt2year_Neptune.ai',
 'q47_study_MLExperiments_tool_nxt2year_WeightsandBiases',
 'q47_study_MLExperiments_tool_nxt2year_Comet.ml',
 'q47_study_MLExperiments_tool_nxt2year_SacredOmniboard',
 'q47_study_MLExperiments_tool_nxt2year_TensorBoard',
 'q47_study_MLExperiments_tool_nxt2year_Guild.ai',
 'q47_study_MLExperiments_tool_nxt2year_Polyaxon',
 'q47_study_MLExperiments_tool_nxt2year_Trains',
 'q47_study_MLExperiments_tool_nxt2year_DominoModelMonitor',
 'q47_study_MLExperiments_tool_nxt2year_None',
 'q47_study_MLExperiments_tool_nxt2year_Other']


# In[19]:


####### Creating fuction which counts the amount of values for q7 by professional_role #######
def counter (dataframe):
    counter_array = []
    for i in range (len(q7)):
        counter_array.append(dataframe[q7[i]].value_counts()[0])
    return counter_array


####### Setting up the data lists #######

#Student Data
q1_program_student = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_student = ['Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student']
q3_student_counter = counter(df[df["q5_professional_role"]=="Student"])


#Data Engineer
q1_program_DataEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataEngineer = ['Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer']
q3_DataEngineer_counter = counter(df[df["q5_professional_role"]=='Data Engineer'])

#Software Engineer
q1_program_SoftwareEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_SoftwareEngineer = ['Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer']
q3_SoftwareEngineer_counter = counter(df[df["q5_professional_role"]=='Software Engineer'])

#Data Scientist
q1_program_DataScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataScientist = ['Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist']
q3_DataScientist_counter = counter(df[df["q5_professional_role"]=='Data Scientist'])

#Data Analyst
q1_program_DataAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataAnalyst = ['Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst']
q3_DataAnalyst_counter = counter(df[df["q5_professional_role"]=='Data Analyst'])

#Research Scientist
q1_program_ResearchScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ResearchScientist = ['Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist']
q3_ResearchScientist_counter = counter(df[df["q5_professional_role"]=='Research Scientist'])

#Other
q1_program_Other = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Other = ['Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other']
q3_Other_counter = counter(df[df["q5_professional_role"]=='Other'])


# Currently not employed
q1_program_NotEmployed = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_NotEmployed = ['Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed']
q3_NotEmployed_counter = counter(df[df["q5_professional_role"]=='Currently not employed'])

# Statistician
q1_program_Statistician = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Statistician = ['Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician']
q3_Statistician_counter = counter(df[df["q5_professional_role"]=='Statistician'])


# Product/Project Manager
q1_program_ProjectManager = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ProjectManager = ['Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager']
q3_ProjectManager_counter = counter(df[df["q5_professional_role"]=='Product/Project Manager'])


# Machine Learning Engineer
q1_program_MLEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_MLEngineer = ['ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer']
q3_MLEngineer_counter = counter(df[df["q5_professional_role"]=='Machine Learning Engineer'])


# Business Analyst
q1_program_BusinessAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_BusinessAnalyst = ['Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst']
q3_BusinessAnalyst_counter = counter(df[df["q5_professional_role"]=='Business Analyst'])



#Merging all lists together
q1_program = q1_program_student+q1_program_DataEngineer+q1_program_SoftwareEngineer+q1_program_DataScientist+q1_program_DataAnalyst+q1_program_ResearchScientist+q1_program_Other+q1_program_NotEmployed+q1_program_Statistician+q1_program_ProjectManager+q1_program_MLEngineer+q1_program_BusinessAnalyst
q2_role=q2_role_student+q2_role_DataEngineer+q2_role_SoftwareEngineer+q2_role_DataScientist+q2_role_DataAnalyst+q2_role_ResearchScientist+q2_role_Other+q2_role_NotEmployed+q2_role_Statistician+q2_role_ProjectManager+q2_role_MLEngineer+q2_role_BusinessAnalyst
q3_count=q3_student_counter+q3_DataEngineer_counter+q3_SoftwareEngineer_counter+q3_DataScientist_counter+q3_DataAnalyst_counter+q3_ResearchScientist_counter+q3_Other_counter+q3_NotEmployed_counter+q3_Statistician_counter+q3_ProjectManager_counter+q3_MLEngineer_counter+q3_BusinessAnalyst_counter

####### initialize data of lists #######
data = {'q1_program':q1_program,
        'q2_role':q2_role,
        'q3_count':q3_count,
       }
 
####### Create DataFrame #######
df_test = pd.DataFrame(data)


####### Plot figure #######
fig = go.Figure()
for gender, group in df_test.groupby("q2_role"):
   fig.add_trace(go.Bar(x = group['q1_program'], y = group['q3_count'], name = gender))
fig.update_layout(barmode="group", plot_bgcolor = "white",title="Most frequent used programming languages by professional role",xaxis_title="programming languages", yaxis_title="count",legend_title_text="Professional role",height=500)
fig.show()


# <h2>Statistical relationship between two categorical variables</h2>

# <h4>V Cramer's Test</h4>

# In[20]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


cramers_v(df['q6_coding_experience'],df['q5_professional_role'])


# <h4>Heatmap visualizing the paircount of two cat variables (only: single answer question columns)</h4>

# In[21]:


def heatmap_paircount_between_cat_variables(dataframe,column1_title,column2_title):
    
    ''' 
        Args: dataframe, first column name as a string, second column name as a string

        Returns: Pairplot heatmap displays the count of appearances between two variables (only: single answer question columns)

        example to evoke: heatmap_paircount_between_cat_variables(df,'q6_coding_experience','q5_professional_role')


    '''
    fig, ax = plt.subplots()
    ct_counts = dataframe.groupby([column1_title, column2_title]).size()
    ct_counts = ct_counts.reset_index(name = 'count')
    ct_counts = ct_counts.pivot(index = column2_title, columns = column1_title, values = 'count')
    heatmap_paircount= sns.heatmap(ct_counts,cmap="YlGnBu")
    return heatmap_paircount

heatmap_paircount_between_cat_variables(df,'q6_coding_experience','q5_professional_role');


# <h4>Correlationmatrix of two cat variables</h4>

# In[22]:


def correlation_between_cat_variables(dataframe,column1_title,column2_title):
    
    
    ''' 
        Args: dataframe, first column name as a string, second column name as a string

        Returns: Correlationmatrix with all the spearman correlations of every single value

        example to evoke: correlation_between_cat_variables(df,'q7_most_freq_used_prgmg_language_C++','q5_professional_role')


    '''

    df_dummies1 = pd.get_dummies(dataframe[column1_title])
    df_dummies2 = pd.get_dummies(dataframe[column2_title])
    df_new = pd.concat([dataframe.drop([column2_title], axis=1), df_dummies1,df_dummies2], axis=1)
    df_new = df_new.iloc[:,1:]
    correlation_matrix = df_new.corr(method = 'spearman')
    return correlation_matrix


# In[23]:


#Evoking the correlation_between_cat_variables function

correlation_between_cat_variables(df,'q7_most_freq_used_prgmg_language_C++','q5_professional_role')


# <h4>Heatmap visualizing correlation of two cat variables</h4>

# In[24]:


def heatmap_between_cat_variables(dataframe,column1_title,column2_title):
    
    '''
    Args: dataframe, first column name as a string, second column name as a string

    Returns: heatmap with all the spearman correlations of every single value

    example to evoke: correlation_between_cat_variables(df,'q7_most_freq_used_prgmg_language_C++','q5_professional_role')

    '''
    df_dummies1 = pd.get_dummies(dataframe[column1_title])
    df_dummies2 = pd.get_dummies(dataframe[column2_title])
    df_new = pd.concat([dataframe.drop([column2_title], axis=1), df_dummies1,df_dummies2], axis=1)
    df_new = df_new.iloc[:,1:]
    plt.figure(figsize = (18,6))
    heatmap = sns.heatmap(df_new.corr(method = 'spearman'), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black',annot= True);
    return heatmap


# <h1>Data Visualization</h1>

# <h3>Who are the respondants of the survey?</h3>

# <h4>Most common nationalities</h4>

# In[25]:


q3_df = df['q3_country'].value_counts()
fig = go.Figure()
fig.add_trace(go.Bar(x = q3_df.index, y = q3_df.values))
fig.update_layout(width=1000, height=700,barmode="group", plot_bgcolor = "white",title="Most Common Nationalities",xaxis_title="countries", yaxis_title="Number of respondents",legend_title_text="Professional role")
fig.show()


# <h5>Commentary</h5>
# 
# <ul>
# <li>India outnumbers over any other countries: 29.2% of all people surveyed are coming from india, 11.2% from the USA</li>
# <li>Nigeria is the most represented african country on Kaggle while</li>
# <li>most partcipants from Europe come from the UK</li>
# </ul>

# <h4>Age and Gender</h4>

# In[26]:


q1_q2_df = df.loc[:, ["q1_age","q2_gender"]].replace(
    {'Prefer not to say':'Divers', 'Nonbinary':"Divers", "Prefer to self-describe": "Divers"}
)
q1_q2_df['q2_gender'].value_counts()


q1_q2_df = q1_q2_df.groupby(["q1_age","q2_gender"]).size().reset_index().rename(columns = {0:"Count"})

fig = go.Figure()
for gender, group in q1_q2_df.groupby("q2_gender"):
   fig.add_trace(go.Bar(x = group['q1_age'], y = group['Count'], name = gender))
fig.update_layout(barmode="group", plot_bgcolor = "white",title="Gender distribution & age ranges participation",xaxis_title="Age (years)", yaxis_title="count",legend_title_text="Gender")
fig.show()


# <h5>Commentary</h5>
# 
# <ul>
# <li>the majority of kaggle members are between the age of 25-29</li>
# <li>age groups 18-21, 22-24 and 25-29 combinely contribute to 51% of total submissions</li>
# <li>Age > 60 make 653 Submissions</li>
# <li>Unbalanced Distribution: women and divers people are very underrepresented on kaggle, which points to the fact that there is an general imbalance between men and women working in the data industry</li>
# <li>summa summarum: kaggle in largely dominated by Indian male students in terms of quantity</li>    
# </ul>

# <h4>Education level distribution</h4>

# In[27]:


trace = go.Pie(labels=['Bachelor’s degree', 'Master’s degree', 'Some college/university without bachelor’s degree',
                       'Doctoral degree', 'I prefer not to answer', "Professional degree", 'No formal education past high school'], 
               values=df['q4_nxt2year_highestEdu'].value_counts().values, 
               title="Education Level Distribution", 
               hoverinfo='percent+value+label', 
               textinfo='percent',
               textposition='inside',
               hole=0.6,
               showlegend=True,
               marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 28)),
                           line=dict(color='#000000',
                                     width=2),
                          )
                  )
fig.update_layout(title="Age distribution by gender",legend_title_text="Gender")
fig=go.Figure(data=[trace])
fig.show()


# <h5>Commentary</h5>
# 
# <ul>
# <li>The participants answered the highest level of formal education attained or planned to attain within the next 2 years.</li>
# <li>More than 40% students are pursuing their Bachelor's Degree</li>
# <li>More than 35% students are planning to attain their Masters's Degree</li>
# <li>Although only 5,5% are aiming for a Phd, the vast majority of 80% of the participants have an academic background</li>    
# </ul>

# <h3>Relationships between professional role (target variable) and survey data</h3>

# <h4>What are the most frequently used programming languages?</h4>

# In[28]:


######################## FIRST PLOT: Most frequent used programming languages #####################
fig = px.histogram(df,q7, title="Most frequent used programming languages")
fig.update_layout(showlegend=False,xaxis_title="programming languages", yaxis_title="count",plot_bgcolor = "white")
fig.show()



######################## SECOND PLOT: Most frequent used programming languages by professional role ###################



####### Creating fuction which counts the amount of values for q7 by professional_role #######
def counter (dataframe):
    counter_array = []
    for i in range (len(q7)):
        counter_array.append(dataframe[q7[i]].value_counts()[0])
    return counter_array


####### Setting up the data lists #######

#Student Data
q1_program_student = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_student = ['Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student']
q3_student_counter = counter(df[df["q5_professional_role"]=="Student"])


#Data Engineer
q1_program_DataEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataEngineer = ['Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer']
q3_DataEngineer_counter = counter(df[df["q5_professional_role"]=='Data Engineer'])

#Software Engineer
q1_program_SoftwareEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_SoftwareEngineer = ['Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer']
q3_SoftwareEngineer_counter = counter(df[df["q5_professional_role"]=='Software Engineer'])

#Data Scientist
q1_program_DataScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataScientist = ['Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist']
q3_DataScientist_counter = counter(df[df["q5_professional_role"]=='Data Scientist'])

#Data Analyst
q1_program_DataAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataAnalyst = ['Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst']
q3_DataAnalyst_counter = counter(df[df["q5_professional_role"]=='Data Analyst'])

#Research Scientist
q1_program_ResearchScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ResearchScientist = ['Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist']
q3_ResearchScientist_counter = counter(df[df["q5_professional_role"]=='Research Scientist'])

#Other
q1_program_Other = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Other = ['Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other']
q3_Other_counter = counter(df[df["q5_professional_role"]=='Other'])


# Currently not employed
q1_program_NotEmployed = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_NotEmployed = ['Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed']
q3_NotEmployed_counter = counter(df[df["q5_professional_role"]=='Currently not employed'])

# Statistician
q1_program_Statistician = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Statistician = ['Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician']
q3_Statistician_counter = counter(df[df["q5_professional_role"]=='Statistician'])


# Product/Project Manager
q1_program_ProjectManager = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ProjectManager = ['Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager']
q3_ProjectManager_counter = counter(df[df["q5_professional_role"]=='Product/Project Manager'])


# Machine Learning Engineer
q1_program_MLEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_MLEngineer = ['ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer']
q3_MLEngineer_counter = counter(df[df["q5_professional_role"]=='Machine Learning Engineer'])


# Business Analyst
q1_program_BusinessAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_BusinessAnalyst = ['Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst']
q3_BusinessAnalyst_counter = counter(df[df["q5_professional_role"]=='Business Analyst'])



#Merging all lists together
q1_program = q1_program_student+q1_program_DataEngineer+q1_program_SoftwareEngineer+q1_program_DataScientist+q1_program_DataAnalyst+q1_program_ResearchScientist+q1_program_Other+q1_program_NotEmployed+q1_program_Statistician+q1_program_ProjectManager+q1_program_MLEngineer+q1_program_BusinessAnalyst
q2_role=q2_role_student+q2_role_DataEngineer+q2_role_SoftwareEngineer+q2_role_DataScientist+q2_role_DataAnalyst+q2_role_ResearchScientist+q2_role_Other+q2_role_NotEmployed+q2_role_Statistician+q2_role_ProjectManager+q2_role_MLEngineer+q2_role_BusinessAnalyst
q3_count=q3_student_counter+q3_DataEngineer_counter+q3_SoftwareEngineer_counter+q3_DataScientist_counter+q3_DataAnalyst_counter+q3_ResearchScientist_counter+q3_Other_counter+q3_NotEmployed_counter+q3_Statistician_counter+q3_ProjectManager_counter+q3_MLEngineer_counter+q3_BusinessAnalyst_counter

####### initialize data of lists #######
data = {'q1_program':q1_program,
        'q2_role':q2_role,
        'q3_count':q3_count,
       }
 
####### Create DataFrame #######
df_test = pd.DataFrame(data)


####### Plot second figure #######
fig = go.Figure()
for gender, group in df_test.groupby("q2_role"):
   fig.add_trace(go.Bar(x = group['q1_program'], y = group['q3_count'], name = gender))
fig.update_layout(barmode="group", plot_bgcolor = "white",title="Most frequent used programming languages by professional role",xaxis_title="programming languages", yaxis_title="count",legend_title_text="Professional role",height=500)
fig.show()


# <h5>Commentary</h5>
# 
# <ul>
# <li>Python and SQL are the most popular jobs in all job roles and students</li>
# <li>R is mainly popular among Data Scientists, Data Analysts and students. Although if you look more closely you will find that R is not very popular among ML Engineers and Data Engineers.</li>
# <li>Bash is more popular among MLOps Engineers also you will find that some other languages also have more popularity against Data Scientists and Data Analysts</li>
# </ul>

# <h3>Statistical visualization</h3>

# <h4>Correlation between professional role and education level</h4>

# In[29]:


#Evoking the heatmap_between_cat_variables function

heatmap_between_cat_variables(df,'q6_coding_experience','q5_professional_role');


# <h5>Commentary</h5>
# 
# <ul>
# <li>heatmap shows the correlation between professional role and coding experience</li>
# <li>We can see a significant positive correlation between the being the group of students and havening < 1 or 1-2 years of coding experience</li>
# <li>As expected, most of the students have very less coding experience</li>
# </ul>

# <h2>Data Preprocessing</h2>

# <h5>Creating df_preprocessed as a copy of df</h5>

# In[30]:


df_preprocessed = df.copy()


# <h4>Preprocessing numerical columns</h4>

# <h5> Yearly compensation: Replacing string values by its corresponding int value (mean)</h5>

# In[31]:


def preprocess_compensation(column):
    def get_mean_compensation(compensation):
        if pd.isnull(compensation):
            return np.nan
        if compensation.startswith('$'):
            compensation = compensation[1:]
        if '>' in compensation:
            split_compensation = compensation.split(' ')
            if len(split_compensation) < 3:
                return np.nan
            compensation = split_compensation[2].replace(',', '')
            return int(float(compensation))
        lower, upper = map(lambda x: float(x.replace(',', '')), compensation.split('-'))
        mean = (lower + upper) / 2
        return int(round(mean))
    return column.apply(get_mean_compensation)



# In[32]:


df_preprocessed['q24_annual_compensation'] = preprocess_compensation(df_preprocessed['q24_yearly_compensation_$USD'])


# In[33]:


df_preprocessed['q24_annual_compensation']
df_preprocessed['q5_professional_role']


# In[34]:


print(df_preprocessed['q24_annual_compensation'].isna().sum())


# <h5>Replace missing values in yearly compensation column with the mean compensation for the corresponding professional role  </h5>

# In[35]:


# Group by professional role and find the mean of the yearly compensation column
mean_compensation_by_role = df_preprocessed.groupby('q5_professional_role')['q24_annual_compensation'].mean()

# Replace missing values with the mean compensation for the corresponding professional role
for role in mean_compensation_by_role.index:
    mean_compensation = mean_compensation_by_role[role]
    df_preprocessed.loc[(df_preprocessed['q24_annual_compensation'].isna()) & (df_preprocessed['q5_professional_role'] == role), 'q24_annual_compensation'] = mean_compensation


# In[36]:


print(df_preprocessed['q24_annual_compensation'].isna().sum())


# In[37]:


df_preprocessed['q5_professional_role'].unique()


# <h5>Dropping rows of non-data job related roles </h5>

# In[38]:


to_drop = ['Student','Other','Currently not employed',np.nan,]

df_preprocessed = df_preprocessed[~df_preprocessed['q5_professional_role'].isin(to_drop)]

df_preprocessed['q5_professional_role'].unique()


# In[39]:


df_preprocessed['q5_professional_role']


# <h5>q6_coding_experience: Replacing string values by its corresponding int value (mean)</h5>

# In[40]:


def preprocess_coding_experience(value):
    if pd.isnull(value):
        return value
    if '<' in value:
        return 1
    if '+' in value:
        return 25
    if 'I have never' in value:
        return 0
    start, end = map(int, value.split(' ')[0].split('-')[:2])
    return (start + end) / 2


# In[41]:


df_preprocessed['q6_coding_experience_years'] = df_preprocessed['q6_coding_experience'].apply(preprocess_coding_experience)


# In[42]:


print(df_preprocessed['q6_coding_experience'].isna().sum())


# In[43]:


df_preprocessed.dropna(subset=['q6_coding_experience'], inplace=True)


# In[44]:


print(df_preprocessed['q6_coding_experience'].isna().sum())


# <h5>Standard scaling numerical variables</h5>

# In[45]:


# Select numerical columns
numerical_cols = df_preprocessed.select_dtypes(include=['float']).columns

# Standardize the numerical columns
scaler = StandardScaler()
df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])


# <h4>Preprocessing categorical columns</h4>
# To preprocess all categorical columns, we use the get_dummies function from pandas. This function creates one-hot encoded columns for each unique category in the input column.

# In[46]:


cat_cols = df_preprocessed.select_dtypes(include='object').columns
df_preprocessed = pd.get_dummies(df_preprocessed, columns=cat_cols, prefix=cat_cols, prefix_sep='_', drop_first=True)


# <h4>Dropping (duplicate/redundant) columns</h4>

# In[47]:


df_preprocessed = df_preprocessed.drop(['duration_seconds'], axis=1)


# In[48]:


df_preprocessed.to_csv('preprocessed_df.csv')


# In[49]:


(df_preprocessed['q24_annual_compensation'].isna().sum()/df_preprocessed['q24_annual_compensation'].shape[0])*100


# In[50]:


(df_preprocessed['q24_annual_compensation'].isna().sum()/df_preprocessed['q24_annual_compensation'].shape[0])*100


# In[51]:


df_preprocessed['q24_annual_compensation'].shape[0]


# In[52]:


df_preprocessed.shape


# In[53]:


df_preprocessed.columns


# In[54]:


df_preprocessed.iloc[:,76:85].columns


# <h2>ML Modelling - Data Analyst or Data Scientist?</h2>

# <h3>Linear Regression</h3>

# In[55]:


# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the dataset into training and testing sets
X = df_preprocessed.drop(columns=['q5_professional_role_DBA/Database Engineer',
       'q5_professional_role_Data Analyst',
       'q5_professional_role_Data Engineer',
       'q5_professional_role_Data Scientist',
       'q5_professional_role_Machine Learning Engineer',
       'q5_professional_role_Product/Project Manager',
       'q5_professional_role_Research Scientist',
       'q5_professional_role_Software Engineer',
       'q5_professional_role_Statistician','q5_professional_role_Data Engineer'])
y = df_preprocessed.loc[:, ['q5_professional_role_Data Analyst',
       'q5_professional_role_Data Scientist']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test into 1D arrays of binary labels
y_train = np.argmax(y_train.values, axis=1)
y_test = np.argmax(y_test.values, axis=1)

# Hyperparameter tuning using grid search
parameters = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
clf = GridSearchCV(LogisticRegression(max_iter=10000), parameters, cv=5)
clf.fit(X_train, y_train)
best_params = clf.best_params_

# Train the logistic regression model with best hyperparameters
model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3>Random Forest</h3>

# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Feature selection using random forest
selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100))
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Hyperparameter tuning using grid search
parameters = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30, None]}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
clf.fit(X_train_selected, y_train)
best_params = clf.best_params_

# Train the random forest model with best hyperparameters
model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# <h3>Gradient Boost Classifier</h3>

# In[57]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Perform grid search using 5-fold cross validation
grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the Gradient Boosting Classifier with the best hyperparameters
model = GradientBoostingClassifier(**best_params)
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# We tested three different models: Logistic Regression, Random Forest, and Gradient Boosting Classifier.
# 
# Logistic Regression is a linear classifier that uses a logistic function to model the probability of the output belonging to a certain class. Here, we used a grid search to tune the hyperparameters of the model, specifically the regularization strength C and the penalty type L2. However, we did not perform any feature selection or scaling on the data, which may have limited the model's performance.
# 
# Next, we used Random Forest, an ensemble method that uses multiple decision trees to make predictions. We also performed feature selection using a Random Forest model to select the most important features for the classification task. We then used a grid search to tune the hyperparameters of the model, including the number of estimators and the maximum depth of the trees. This resulted in a slight improvement in accuracy compared to Logistic Regression.
# 
# Finally, we used Gradient Boosting Classifier, another ensemble method that sequentially adds models to correct errors made by previous models. We used the same features selected by the Random Forest model and did not perform any additional hyperparameter tuning. This resulted in the highest accuracy of the three models tested.
# 
# Overall, we can see that the performance of the models improved as we incorporated more advanced techniques such as feature selection and hyperparameter tuning. The Gradient Boosting Classifier performed the best, likely due to its ability to correct errors made by previous models and its use of a strong regularization term. In any case, an accuracy of around 0.79 is not bad, and the model seems to be performing reasonably well on the given dataset.

import streamlit as st
import pandas as pd
import numpy as np

st.title('Model deployment of data jobs')

streamlit run DataJobProject.py
