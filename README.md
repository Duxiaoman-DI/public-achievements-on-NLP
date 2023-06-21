# public-achievements-on-NLP    
数智化转型是金融行业的重要趋势，而NLP等前沿技术是实现转型的关键手段。度小满的AI-lab团队在NLP技术上有着多年的深入研究和探索，尤其在金融场景服务上积累了丰富的技术经验和人才资源，并取得了多项突破。2021年，在微软举办的MS MARCO 比赛中的文档排序任务中，度小满NLP团队排名第一并刷新纪录；团队研发的轩辕 (XuanYuan) 预训练模型也在CLUE分类任务中排名第一。2022年，度小满在自然语言处理研究的重点领域——文本匹配、知识表示、预训练语言模型——的三篇科研论文被NLP三大顶级会议之一的EMNLP接收。度小满AI-lab团队有志于持续推动NLP技术在金融场景的深入应用。    

**比赛与奖项**     
* 2021 MS MARCO挑战赛：第一名
* 2022 CLUE 1.1 分类任务：第一名
* 2022 OGBL-wikikg2：第一名


**顶会论文**   
* ACL
  * (2023) Pre-trained Personalized Review Summarization with  Effective Salience Estimation
  * (2023) A^2-Former: Adaptive Attention for Sparse-Based Long-Sequence Transformer
* EMNLP
  * (2022) Instance-Guided Prompt Learning for Few-Shot Text Matching
  * (2022) Transition-based Knowledge Graph Embedding with Synthetic Relation
  * (2022) ExpertPLM: Pre-training Expert Representation for Expert Finding 
* CIKM    
  * (2022) ExpertBert: Pretraining Expert Finding
  * (2022) Efficient Non-sampling Expert Finding
  * (2022) DeepVT:Deep View-Temporal Interaction Network for News Recommendation
  * (2021) DML: Dynamic Multi-Granularity Learning for BERT-Based Document Reranking
* ACM MM
  * (2021) Position-Augmented Transformers with Entity-Aligned Mesh for TextVQA
* WWWW   
  * (2021) Combining Explicit Entity Graph with Implicit Text Information for News Recommendation


## 比赛与奖项   

### 2021 MS MARCO挑战赛   
**比赛介绍**  MS MARCO挑战赛是自然语言处理NLP领域的权威比赛，基于微软构建的大规模英文阅读理数据集 MARCO，需要参赛者为用户输入的问题找寻到最贴切、最需要的答案，并对答案进行排序。   
该赛事吸引了谷歌、韩国三星AI研究院、斯坦福大学、清华大学等科技巨头、顶尖学术机构参与，竞争十分激烈，挑战赛的难点在于对于无法在文档中直接找到答案的问题，需要AI有能够像人类一样阅读和理解文档，生成正确答案的能力。    
**参赛结果**  我们首次提出了DML文本排序算法，通过自主研发的自适应预训练语言模型对query（用户搜索的真实问题）和document文本进行深度理解，利用了数十万数据来训练模型，经过召回、重排等多个阶段，给出最终排序。   
该模型不仅以0.416的eval分数大幅领先其他团队，包括三星、微软、谷歌、斯坦福、清华大学等一众参赛者，并在第一名的位置维持了一个多月时间。    

### 2022 CLUE 1.1 分类任务   
**比赛介绍**  CLUE是中文语言理解领域最具权威性的测评基准之一，涵盖了文本相似度、分类、阅读理解共10项语义分析和理解类子任务。其中，分类任务需要解决6个问题，例如传统图像分类，文本匹配，关键词分类等等，能够全方面衡量模型性能。    
该榜单竞争激烈，几乎是业内兵家必争之地，例如快手搜索、优图实验室& 腾讯云等等研究机构也都提交了比赛方案。      
**参赛结果**  我们首次提出了轩辕预训练大模型，“轩辕”是基于Transformer架构的预训练语言模型，涵盖了金融、新闻、百科、网页等多领域大规模数据。因此，该模型“内含”的数据更全面，更丰富，面向的领域更加广泛。传统预训练模型采取“训练-反馈”模式，我们在训练“轩辕”的时候细化了这一过程，引入了任务相关的数据，融合不同粒度不同层级的交互信息，从而改进了传统训练模式。     
最终，轩辕预训练大模型在CLUE1.1分类任务中“力压群雄”获得了排名第一的好成绩，距离人类“表现”仅差3.38分。

### 2022 OGBL-wikikg2 挑战赛  
**比赛介绍**   。      
**参赛结果**   。


## 顶会论文  

### ACL 2023: Pre-trained Personalized Review Summarization with  Effective Salience Estimation    
### ACL 2023: A^2-Former: Adaptive Attention for Sparse-Based Long-Sequence Transformer  
### EMNLP 2022: Instance-Guided Prompt Learning for Few-Shot Text Matching   
### EMNLP 2022: Transition-based Knowledge Graph Embedding with Synthetic Relation   
### EMNLP 2022: ExpertPLM: Pre-training Expert Representation for Expert Finding   
### CIKM 2022: ExpertBert: Pretraining Expert Finding  
### CIKM 2022: Efficient Non-sampling Expert Finding
### CIKM 2022: DeepVT:Deep View-Temporal Interaction Network for News Recommendation
### CIKM 2021: DML: Dynamic Multi-Granularity Learning for BERT-Based Document Reranking 
### ACM MM 2021: Position-Augmented Transformers with Entity-Aligned Mesh for TextVQA
### WWW 2021: Combining Explicit Entity Graph with Implicit Text Information for News Recommendation 

