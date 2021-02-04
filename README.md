# FedEEG

https://figshare.com/articles/journal_contribution/Work/13708612

## Abstract

EEG technology and analysis offers a great opportunity for developing Brain-Computer-Interface. Machine learning shows great potential in this subject of study, however, the subject of study is of great sensitivity. BCI or EEG analysis has been under heavy discussion and scrutiny with respect to privacy concerns and ethics. Leakage of data, which may include personal emotions and sentiments as well as ethical concerns with misrepresentation of learnt data by ML models. Previous studies on EEG analysis have focused on decoding emotions, hand-movements and thought representation. However, these studies do not focus on contributor privacy and therefore this work attempts to provide a proof-of-concept for employing Federated Learning for conserving the same. The analysis provided employs EEG data collected for Hand Movement Classification from 4 different clients. The results demonstrate the effectiveness of Federated Learning as an aggregating tool for EEG analysis. Achieving an accuracy of 57.67% which is comparable to 61.83% when utilizing the complete dataset. The study focuses on privacy and the employment of federated learning as a means of overcoming it.

## 1.  Introduction

With much research and development in the field of EEG analysis and brain-computer interface, the rise of ethical and data privacy concerns of participating clients become an integral part of this discipline. EEG analysis is a powerful tool, producing some intricate ethical concerns. Unconcensual commercialization of EEG waves for sentiment or thought analysis are areas of ethical concern. The idea of thought analysis and reading could lead to privacy breach and in an extreme case even towards thought policing. EEG falls under medical data and therefore carries with it the repercussions of dealing with sensitive patient data. There have been multiple cases of unintentional privacy breaches in public datasets due to de-anonymization attacks [1][2]. Therefore privacy and ethical concerns are an integral part of the future development of this discipline and federated learning offers a convenient yet robust solution to it.

Federated Learning, a powerful technology for multi-client learning under a centralized or decentralized hub, can become a privacy conserving tool for EEG analysis. The collaborative and privacy conserving aspect of this algorithm can allow medical data to be freely acted upon. This would open windows to new opportunities as additional information/data prove to boost ML performance in EEG analysis [3]. The aspect of allowing collaborative learning will also prove to boost research in the field, with data being easily available for use. Also, with medical hospitals governing the information flow involved with this data, it can choose to become a body maintaining the flow governance in this information transaction.

The problems currently faced by EEG data analysis and BCI research are summarized below, as well as, the reasons why federated learning may offer a great solution to the same.

* Some of the concerns at the moment,
* Ethical concerns against data analysis for non-consented tasks.
* Privacy breach of individuals participating in data collection.
* Risk of de-anonymization attacks on large publicly available datasets.

Federated Learning offers a convenient solution to the above issues,
* Privacy conservation mechanism due to collaborative learning and Federated-Averaging algorithm.
* The opportunity of crowdsourcing collaborative learning, allowing the aggregate model to converge over a large sample space and be more generalized.
* Allows hospitals to implement flow governance, thereby, preventing privacy breach even through Federated Learning.
* Utilization of Secure Aggregation and/or Differential Privacy may offer further security assurance.

The goal of this study is to bring to light the ethical and privacy concerns held by EEG analysis. The reasons why large datasets are still unavailable in the field and finally offer an alternative to the aforementioned issues. EEG data can be learnt in a collaborative and privacy conserving manner is presented further in the paper and a detailed analysis has been organized to study the effects of this methodology on model performances.

The paper is structured into 5 main sections. In Section 2, a survey has been done on current EEG analysis work, federated learning and works combining the two. This section offers a brief introduction to the research done in these fields and builds a base examination for the combination of both these technologies in section 3. Here, the proposed methodology is formalized and presented. In section 4, the empirical results and observations are described alongside important technical details of the experiments are noted. Section 5 offers a conclusion to this study and prospective future work.

## 2. Related Work:

### 2.1 EEG Analysis:

EEG is a medical imaging technique that reads electrical activity generated by brain structures. It measures voltage fluctuations resulting from ionic current flows within the neurons of the brain. The data collected offers great potential in learning how data is presented within our brain and performing actions based on the same. EEG data analysis has become a topic of great interest in recent years. Emotion recognition, hand movement classification and thought representation are some of the tasks EEG analytics are capable of. Emotion recognition using EEG data may be of great use for paralyzed or semi-paralyzed patients. Multiple machine learning algorithms have been employed to carry out this task. Support Vector Machines is one of the most common methodologies used for recognition [4][5][6]. Among other machine learning algorithms, K-Nearest Neighbour and Linear Discriminant Analysis have been some of the other prominent technologies[4]. On the other hand, the use of RNN was utilized for neurodegeneration prognosis [7]. This methodology is further discussed for Alzheimer's recognition and brain state inference [8][9]. These work discuss the use of EEG for different tasks and their potential use-cases/motivations. Recurrent neural networks offer a generalized learning model for complex tasks and therefore is a robust learning model.

### 2.2 Federated Learning:

Konecnˇy et al. proposed Federated Learning as a framework for distributed learning while securing data privacy [10]. Over the last few years, numerous Federated Learning frameworks and architectures have been proposed for various problem statements, including computer vision [11], medical image analysis [12] as well as linguistic data analysis. It offers a convenient and easy to implement a solution to a once-given problem of client privacy for data collection. At the same time, looking at some other trivial solutions such name anonymization, critical information withholding, etc. not only reduces the final performance of the model but also takes massive computational/manual effort to perform. Such pre-processing steps are a hindrance which prevented the collection of massive datasets, spanning across institutions and individuals. However, federated learning offers a convenient workaround to this problem. Federated DNN offers to solve such collaborative learning problems in an optimal manner while keeping intact the privacy of the users involved.

### 2.3 Federated Learning and EEG Analysis

Both the above technologies have also been studied in a co-operative manner. A genetic algorithm-based federated learning model was proposed for detecting brain-activation using stimuli [13]. Another study discussing privacy-preserving demands and challenges employs heterogeneous federated learning approach for EEG data learning [14]. A discussion on the lack of large-datasets for BCI tasks has been discussed in Ce et al. [15]. This study proposes federated transfer learning to improve performance of federated learning on EEG data. The above work is analysed and a methodology for hand movement detection is proposed for collaborative learning.

## 3. Methodology:

FedEEG employs Federated Averaging, to classify hand movement from EEG signals collected form individual clients. The aim is to produce an efficient model, which can be trained in a distributed fashion while conserving privacy. As observed above the use of RNN allows models to learn complex features and therefore the same is employed in our task. Unlike fully connected dense layers, the use of recurrent neural networks allows the model to learn sequential patterns. For federated model training, the weight updates are computed and communicated to the central server at the end of local training. This allows for faster convergence of the federated model. However, the setting we propose our methodology under is centralized federated learning. The use of centralized federated learning allows medical bodies to practice flow governance and further ensuring privacy conservation.

The federated averaging algorithm begins by initializing a central model with trainable parameters or weights, which is then distributed across all client devices. Now given the initial model parameters from the central server, each device accesses its own local data, which consists of hand movement labels and their respective EEG wave readings. This data is then divided into fixed batches and each sample batch, gradients are computed and applied to update the local parameters. After iterating over all batches, the updated parameter values are sent to the central server for aggregation and the whole process is repeated until satisfaction. 

```

Procedure FEDERATED TRAIN:

Initialize Θ(0)
for t ∈ {1, …, T} do
	U(t) ← (random set of m clients)
	for each client[k] ∈ U(t) in parallel do
		w[k](t+1) ← ClientUpdate(k, w(t))
	w(t+1) ← summation( w[k](t+1)/n,  for k=1 to K)
		
Procedure CLIENT UPDATE (client[k]): 
B ← (D[k] into batches of size B_size) 
for each local epoch i from 1 to E do 
for batch b ∈ B do 
w ← w − η∇l(w; b) 
return w to server

```

The federated averaging algorithm allows us to converge over this distributed EEG data. The ability of federated learning to conserve privacy while allowing a collaborative learning approach may allow crowdsourcing EEG data from enthusiasts and hobbyists. The above algorithm is simple in complexity and easily deployable at the same time does not require heavy learning resources. Smaller dataset samples also mean the learning progression is quicker and more convenient. The proposition of Federated Averaging clubbed with EEG data achieves a comparable performance to baseline learning and therefore can be utilized for the aforementioned learning task.

## 4. Experimental Results:

### 4.1 Implementation Details:

FedEEG has been deployed using the Federated Averaging algorithm. This algorithm has a faster convergence rate and higher generalization factor than stochastic federated learning. The ability of this methodology to ensure privacy is a major advantage and it can further be coupled with differential privacy, homomorphic encryption to enhance security.

Recurrent Neural Networks on the other hand offer connections between nodes form a directed graph along a temporal sequence, thereby allowing sequence representation and therefore learning. The use of this concept, specifically LSTMs (Long Short Term Memory networks), allowed the aggregate model to represent EEG waves in a sequential learnable manner. This allowed the EEG data to be used for hand movement classification.

The FedEEG model is implemented on TensorFlow. On client updates, model parameters are
trained with the Adam optimizer, with initial learning rate 0.001. In the reported experiments, we fix the batch size to 12 for learning. An 80:20 split has been made for training testing samples, allowing us to measure the learnability of the proposed model.

### 4.2 Dataset:

The EEG data is taken from, Fabricio Torquato, “EEG data from hands movement ” [15]. The data is collected from 4 different users over 3 different hand movement classes. During each data acquisition cycles, the participant was exposed to the images depicting voluntary motor actions, namely: a right arrow that would represent motor action in the right direction, a left arrow that would represent motor action from the left direction and a circle that would represent no motor action. 
The dataset consists of 2,880 samples per user, spanning over the three categories. A split of 80:20 has been made to create training and validation samples, these samples have been randomly distributed.

### 4.3 Results:

The use of Federated Averaging reduced accuracy by 4.16% and an increase in the loss by 0.12. These results are substantial in the belief that federated learning can be used to converge models in a distributed-privacy-conserving fashion. The detailed client results have been presented in Table 1 and their learning progression Figure 1.


![fig 1](/images/acc.png)


**Figure 1:** The above image shows the progression of the accuracy of individual clients after every aggregation. The image clearly shows the learnable capacity of FedEEG.

**Table 1:** The table shows the performance of different clients over the course of learning. The table shows client-wise accuracy and loss values after each aggregation step.

client_id_0_loss | client_id_0_acc | client_id_1_loss | client_id_1_acc | client_id_2_loss | client_id_2_acc | client_id_3_loss | client_id_3_acc
---|---|---|---|---|---|---|---
0.9628694405158361 | 0.515625 | 0.9564681400855383 | 0.5225694179534912 | 1.0691524545351665 | 0.4166666567325592 | 1.114998849729697 | 0.3489583432674408
0.9289976650228103 | 0.5381944179534912 | 0.9217653137942156 | 0.515625 | 1.0365764796733856 | 0.4635416567325592 | 1.3993440320094426 | 0.4236111044883728
0.9374355437854925 | 0.5190972089767456 | 0.8785319017867247 | 0.5729166865348816 | 1.0338255328436692 | 0.4548611044883728 | 1.1058493840197723 | 0.421875
0.8807635648796955 | 0.59375 | 1.0283631086349487 | 0.5416666865348816 | 1.0530484008292358 | 0.4427083432674408 | 0.9789415535827478 | 0.5416666865348816
0.9172444740931193 | 0.5902777910232544 | 0.8444005412360033 | 0.5972222089767456 | 1.0650703931848209 | 0.4513888955116272 | 0.9781554614504179 | 0.5190972089767456

Find the code and the details related to the model on the GitHub repository (https://github.com/AmanPriyanshu/FedEEG/).

## 5. Conclusion:

Federated Learning provides an efficient platform for collaborative learning to detect hand movement using EEG waves. It not only negates the requirement of a single server-based training but also allows for faster and continuous learning over multiple users. This ability can be used by hospitals to produce high-quality models trained on larger data samples. The ability to conserve privacy also allows a gateway of opportunities. One such being the ability is to allow crowdsourcing data to volunteers, enthusiasts and hobbyists while ensuring their privacy. This would allow models to learn on large data samples and be more generalized.

EEG analysis and BCI is a growing field in machine learning and therefore the lack of larger datasets and the limitations on the pre-existing samples are with holding the growth of the field, privacy enabled machine learning offers a methodology to solve this issue. Federated Learning a subset of algorithms falling under this, offers a convenient, easy to implement solution towards hand-movement learning.

## Citations:
```
[1]: A. Narayanan, V. Shmatikov: “Robust De-anonymization of Large Sparse Datasets”
[2]: Z. Sun, R. Schuster, V. Shmatikov: “De-Anonymizing Text by Fingerprinting Language Generation”
[3]: L.A.W. Gemein, R.T. Schirrmeister, P. Chrabąszcz, D. Wilson, J. Boedecker, A. Schulze-Bonhage, F. Hutter, T. Ball: “Machine-learning-based diagnostics of EEG pathology”
[4]: S.M. Alarcão; M.J. Fonseca: “Emotions Recognition Using EEG Signals: A Survey”
[5]: Y. Velchev, S. Radeva, S. Sokolov and D. Radev: "Automated estimation of human emotion from EEG using statistical features and SVM”
[6]: R. Mehmood and H.J. Lee: "Towards emotion recognition of EEG brain signals using Hjorth parameters and SVM"
[7]: G. Ruffini, D. Ibañez, M. Castellano, S. Dunne, A. Soria-Frisch: “EEG-driven RNN Classification for Prognosis of Neurodegeneration in At-Risk Patients”
[8]: S. Patnaik, L. Moharkar; A. Chaudhari: “Deep RNN learning for EEG based functional brain state inference”
[9]: A.A. Petrosiana, D.V. Prokhorovb, W. Lajara-Nansona, R.B. Schiffera: “Recurrent neural network-based approach for early recognition of Alzheimer's disease in EEG”
[10]: J. Koneˇcn´y, H.B. McMahan, D. Ramage, P. Richt´arik: “Federated Optimization: Distributed Machine Learning for On-Device Intelligence”
[11]: L.T. Phong, Y. Aono, T. Hayashi, L. Wang, S. Moriai: “Privacy-Preserving Deep Learning via Additively Homomorphic Encryption”
[12]: M.J. Sheller, G.A. Reina, B. Edwards, J. Martin, S. Bakas: “Multi-Institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation”
[13]: G. Szegedi, P. Kiss, T. Horváth: “Evolutionary federated learning on EEG-data”
[14]: D. Gao, C. Ju, X. Wei, Y. Liu, T. Chen, Q. Yang: “HHHFL: Hierarchical Heterogeneous Horizontal Federated Learning for Electroencephalography”
[15]: F. Torquato, “EEG data from hands movement ”
```
