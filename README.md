# GraphCDA
This code is the implementation of GraphCDA
<br>
<br>
<br>
[Dai Qiguo(导师)](https://www.dlnu.edu.cn/comd/szdw66/rgzn/116969.html), [**Liu Ziqiang**](https://orcid.org/0000-0002-0002-4569) et al. GraphCDA: a hybrid graph representation learning framework based on GCN and GAT for predicting disease-associated circRNAs[J]. *Briefings in Bioinformatics*, 2022,23(5):bbac379.
<br>
中科院 SCI分区（2020基础版）  **生物学 1 区**，**JCR Q1**，影响因子：**13.994**，**TOP** 期刊 
<br>
<br>
<br>

#### Requirements

* python (tested on version 3.7.11)  
* pytorch (tested on version 1.6.0)  
* torch-geometric (tested on version 2.0.2)  
* numpy (tested on version 1.21.2)  
* scikit-learn(tested on version 1.0.2)  

#### Quick start

To reproduce our results:  
Run code\main.py to RUN GraphCDA.  

#### Folder

* code: Model code of GraphCDA.  
* datasets: Data required by GraphCDA.  
* otherdatasets: several public databases
* results: Results of GraphCDA run.




#### Data description
* all_circrna_disease_pairs.csv: all pairs of circRNAs and diseases  
* c_c.csv: circRNA integrated similarity  
* c_d.csv: circRNA-disease association matrix    
* circname.txt: list of circRNA names  
* d_d.csv: disease integrated similarity   
* disease semantic similarity.csv: disease semantic similarity  
* disname.txt: list of disease names  
* funicircRNA.csv: circRNA functional similarity  

#### Contacts

If you have any questions or comments, please feel free to email Ziqiang Liu(liuzq_dlmu@163.com) 



