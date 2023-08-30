# public-achievements-on-NLP    
数智化转型是金融行业的重要趋势，而NLP等前沿技术是实现转型的关键手段。度小满的AI-lab团队在NLP技术上有着多年的深入研究和探索，尤其在金融场景服务上积累了丰富的技术经验和人才资源，并取得了多项突破。2021年，在微软举办的MS MARCO 比赛中的文档排序任务中，度小满NLP团队排名第一并刷新纪录；团队研发的轩辕 (XuanYuan) 预训练模型也在CLUE分类任务中排名第一。2022年，度小满在自然语言处理研究的重点领域——文本匹配、知识表示、预训练语言模型——的三篇科研论文被NLP三大顶级会议之一的EMNLP接收。度小满AI-lab团队有志于持续推动NLP技术在金融场景的深入应用。    

**比赛与奖项**     
* [2021 MS MARCO挑战赛：第一名](#msmarco)
* [2022 CLUE 1.1 分类任务：第一名](#clue)
* [2022 OGBL-wikikg2：第一名](#ogbl)


**顶会论文**   
* ACL
  * [(2023) Pre-trained Personalized Review Summarization with  Effective Salience Estimation](#aclpretrain)
  * [(2023) Adaptive Attention for Sparse-Based Long-Sequence Transformer](#aclformer)
* IJCAI
  * [(FinLLM 2023) CGCE: A Chinese Generative Chat Evaluation Benchmark for General and Financial Domains](#ijcaicgce)
* EMNLP
  * [(2022) Instance-Guided Prompt Learning for Few-Shot Text Matching](#instance)
  * [(2022) Transition-based Knowledge Graph Embedding with Synthetic Relation](#transition)
  * [(2022) ExpertPLM: Pre-training Expert Representation for Expert Finding ](#expertplm)
* CIKM
  * [(2023) XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters](#ijcaixuanyuan)
  * [(2022) ExpertBert: Pretraining Expert Finding](#expertpretrain)
  * [(2022) Efficient Non-sampling Expert Finding](#nosample)
  * [(2022) DeepVT:Deep View-Temporal Interaction Network for News Recommendation](#deepvt)
  * [(2021) DML: Dynamic Multi-Granularity Learning for BERT-Based Document Reranking](#dml)
* ACM MM
  * [(2021) Position-Augmented Transformers with Entity-Aligned Mesh for TextVQA](#textvqa)
* WWWW   
  * [(2021) Combining Explicit Entity Graph with Implicit Text Information for News Recommendation](#news)


## 比赛与奖项   

### <span id='msmarco'>2021 MS MARCO挑战赛</span>   
**比赛介绍**  MS MARCO挑战赛是自然语言处理NLP领域的权威比赛，基于微软构建的大规模英文阅读理数据集 MARCO，需要参赛者为用户输入的问题找寻到最贴切、最需要的答案，并对答案进行排序。   
该赛事吸引了谷歌、韩国三星AI研究院、斯坦福大学、清华大学等科技巨头、顶尖学术机构参与，竞争十分激烈，挑战赛的难点在于对于无法在文档中直接找到答案的问题，需要AI有能够像人类一样阅读和理解文档，生成正确答案的能力。    
**参赛结果**  我们首次提出了DML文本排序算法，通过自主研发的自适应预训练语言模型对query（用户搜索的真实问题）和document文本进行深度理解，利用了数十万数据来训练模型，经过召回、重排等多个阶段，给出最终排序。   
该模型不仅以0.416的eval分数大幅领先其他团队，包括三星、微软、谷歌、斯坦福、清华大学等一众参赛者，并在第一名的位置维持了一个多月时间。    
<br/>


### <span id='clue'>2022 CLUE 1.1 分类任务</span>   
**比赛介绍**  CLUE是中文语言理解领域最具权威性的测评基准之一，涵盖了文本相似度、分类、阅读理解共10项语义分析和理解类子任务。其中，分类任务需要解决6个问题，例如传统图像分类，文本匹配，关键词分类等等，能够全方面衡量模型性能。    
该榜单竞争激烈，几乎是业内兵家必争之地，例如快手搜索、优图实验室& 腾讯云等等研究机构也都提交了比赛方案。      
**参赛结果**  我们首次提出了轩辕预训练大模型，“轩辕”是基于Transformer架构的预训练语言模型，涵盖了金融、新闻、百科、网页等多领域大规模数据。因此，该模型“内含”的数据更全面，更丰富，面向的领域更加广泛。传统预训练模型采取“训练-反馈”模式，我们在训练“轩辕”的时候细化了这一过程，引入了任务相关的数据，融合不同粒度不同层级的交互信息，从而改进了传统训练模式。     
最终，轩辕预训练大模型在CLUE1.1分类任务中“力压群雄”获得了排名第一的好成绩，距离人类“表现”仅差3.38分。   
<br/>  


### <span id='ogbl'>2022 OGBL-wikikg2 挑战赛</span>  
**比赛介绍**   OGB是斯坦福大学发布的国际知识图谱基准数据集，也是图神经网络领域最权威、最具挑战性的「竞技场」，每年都有众多顶级研究机构和企业前来参赛。本次AI-lab参赛的是难度颇高的ogbl-wikikg2，该榜单数据来源于Wikidata知识库，涵盖现实世界约250万个实体之间的500多种不同关系，构成了1700多万个事实三元组。       
**参赛结果**   我们在数据处理和算法优化等方面进行了数以千次的实验后，提出了两实体间多样化的关系合成模式，形成了TranS模型。    
新模型TranS，突破了基于翻译的知识表示学习中传统分数模式，通过实体节点间关系向量的合成与推理提升复杂场景下知识图谱建模的能力。同时，在同一实体对的不同关系表示上，效果远超TransE、InterHT、TripleRE、TransH、PairRE等现有方法。   
最终，TranS模型在OGBL-Wikikg2基准数据集刷新最高记录，碾压Meta（原Facebook）AI实验室FAIR、蒙特利尔Mila实验室等一众国内外顶级AI科研机构，创造了KGE算法新纪录。      
![trans图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/KGEcompetition.png)      
<br/>  




## 顶会论文  

### <span id='aclpretrain'>ACL 2023: Pre-trained Personalized Review Summarization with  Effective Salience Estimation</span>    
待更新    
<br/>  

### <span id='aclformer'>ACL 2023: A^2-Former: Adaptive Attention for Sparse-Based Long-Sequence Transformer</span>    
待更新    
<br/>  

### <span id='instance'>EMNLP 2022: Instance-Guided Prompt Learning for Few-Shot Text Matching</span>   
**论文简介**  少样本文本匹配是自然语言处理中一种重要的基础任务，它主要用于在少量样本情况下确定两段文本的语义是否相同。其主要设计模式是将文本匹配重新转换为预训练任务，并在所有实例中使用统一的提示信息，但这种模式并没有考虑到提示信息和实例之间的联系。所以我们认为**实例和提示之间动态增强的相关性是必要的**，因为单一的固定的提示信息并不能充分适应推理中的所有不同实例。为此我们提出了IGATE模型用于少样本的文本匹配，它是一种新颖的且可以即插即用的提示学习方法。**IGATE模型中的gate机制应用于嵌入和PLM编码器之间，利用实例的语义来调节gate对提示信息的影响**。实验结果表明，IGATE在MRPC和QQP数据集上实现了SOTA性能并优于之前最好的基线模型。     
**论文链接**  [论文](https://aclanthology.org/2022.findings-emnlp.285/)    
![instance图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/instance-guided.png)   
<br/>   

### <span id='transition'>EMNLP 2022: Transition-based Knowledge Graph Embedding with Synthetic Relation</span>   
**论文简介**  在自然语言处理任务中如何将知识的关联关系引入到模型中是一项具有挑战性的任务，同时也是KG其他下游任务的基础，如知识问答、知识匹配和知识推理等。虽然预训练模型中已经暗涵各类常识知识，但是如何显式地表示知识中各元素的关联仍然是十分重要的问题。**所以我们提出并构建了新的关系嵌入模式，即构建三段式的关系表示并使得头尾实体的差值近似于该表示**。具体来说，三段式合成关系表示中的两部分先分别与头尾实体进行交互并产生新的向量表示，最后将新的三段式关系表示进行合成形成最终的关系嵌入用于模型的训练。实验结果表明，我们的模型可以在相似参数量的情况下有效提升模型性能。    
**论文链接**  [论文](https://aclanthology.org/2022.findings-emnlp.86/)      
![trasitionbased图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/transitionbasedkgd.png)   
<br/>   


### <span id='expertplm'>EMNLP 2022: ExpertPLM: Pre-training Expert Representation for Expert Finding</span>   
**论文简介**  本文是在CIKM 2022论文ExperBert的基础上，进一步挖掘如何利用用户的历史文本数据（如搜索内容，回答问题等）对用户进行个性化预训练表征。虽然ExpertBert能够保持预训练与下游任务一致性，然而学到的用户表示局限与某一类下游任务。因此本文提出的Expert PLM与下游任务解耦，旨在利用预训练语言模型PLMs学习更加通用和准确的个性化用户表示。首先将用户的所有历史行为进行聚合，得到该用户的预训练语料。此处我们**不仅聚合每条历史行为的文本内容，而且将历史行为的用户特征融合到输入中，来表示该用户对该条记录的影响**，以社区问答为例，用户的每个历史回答收到的投票数可以显示出用户回答该问题的能力，在实际业务中，我们将每条历史记录的时间、位置等用户个性化信息，融合到预训练中。这样的预训练语料构造方式相比仅仅利用文本内容，能够体现出当前用户的个性化特性。**此外，本文提出一种融合用户画像信息的预训练方式，在掩码预训练模型(MLM)的基础上，同时对用户画像进行预测，这样能够进一步提升模型在预训练过程中学到更多个性化表征**。    
在社区问答专家发现的下游任务中，ExpertPLM模型能够在多个公开数据集能够显著超越基线算法，实现优异的性能。    
**论文链接**  [论文](https://aclanthology.org/2022.findings-emnlp.74/)
![expertplm图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/expertplmpretrain.png)   
<br/>    

### <span id='expertpretain'>CIKM 2022: ExpertBert: Pretraining Expert Finding</span>  
**论文简介**  本文主要研究如何利用用户的历史文本数据（如搜索内容，回答问题等）学习用户表示。近年来预训练技术在自然语言处理领域取得了重大进展，也开始用于用户建模任务。 然而，大多数预训练模型（PLM）是基于语料或者文档粒度，这与下游用户粒度建模任务并不一致，因此有必要设计一种更有效的预训练框架能够在用户粒度进行建模。本文提出了一种简单有效的用户级别的预训练模型，命名为ExpertBert，在预训练阶段有效地在统一了文本表示、用户建模和下游任务。具体来说，首先将每个用户的所有历史文本进行聚合，作为用户粒度的语料，用来进行后续预训练，相比对于单条文本，这样历史聚合的方式能够使模型在预训练过程中学习用户本身的语义表征。此外，本文设计了一种标签增强的掩码语言模型(MLM)，将下游的用户建模的监督标签融合到预训练的权重学习中，进一步使预训练更接近下游用户粒度的建模任务。在社区问答专家发现公开数据集的实验结果表明ExpertBert能够超越基线算法，实现优异的性能。          
**论文链接**  [论文](https://dl.acm.org/doi/abs/10.1145/3511808.3557597)   
![expert图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/ExpertBert%20%20Pretraining%20Expert%20Finding.png)   
<br/>    


### <span id='nosample'>CIKM 2022: Efficient Non-sampling Expert Finding</span>    
**论文简介**  本文主要研究如何利用非采样技术上(Non-sampling)对用户历史行为序列建模。目前大部分的用户建模普遍依赖负采样技术来高效训练，然而负采样由于对不同的采样方法和数量高度敏感，会导致模型不稳健，同时也会损失大量有用信息。近年来，非采样技术受到研究者的关注，相比于负采样，非采样能够有效利用全部数据，保证模型的稳定性。目前在推荐系统领域已有若干非采样的研究，有效简化了非采样全数据建模的复杂度问题， 但这些方法往往关注基于用户和商品的ID特征的单一的CTR场景，无法直接扩展到用户交互行为复杂、以及一些冷启动的任务中。本文提出一种基于非采样的用于复杂交互行为的用户表示模型，命名为ENEF。首先对用户的历史行为（如文本内容），采用相应编码器进行特征表示，并且针对用户复杂交互行为的场景，精心设计了多任务损失函数，之后通过高效的全数据优化方法，能够在不进行负采样的情况下学习模型的权重，对数据的利用更加充分，用户的表示更加精准。通过在开源社区问答用户建模数据集上进行大量实验，结果表明ENEF在模型性能和训练效率上均达到最优效果，证明了模型在用户建模的有效性。。      
**论文链接**  [论文](https://dl.acm.org/doi/abs/10.1145/3511808.3557592)     
<br/>    

### <span id='deepvt'>CIKM 2022: DeepVT:Deep View-Temporal Interaction Network for News Recommendation</span>     
**论文简介**  用户行为与兴趣复杂多变，本文提出深度视图时间交互网络来准确学习用户表示。前人的工作大多只将项目级表示直接应用于用户建模中，视图级的信息往往被压缩为稠密的向量，这使得不同浏览项目中的不同视图无法有效的融合。在本文中，我们主要关注于用户建模的视图级信息，并提出用于推荐的深度视图时间交互网络。它主要包括两个部分，即2D半因果卷积神经网络（SC-CNN）和多算子注意力（MoA）。SC-CNN可以同时高效地合成视图级别的交互信息和项目级别的时间信息。MoA在自注意力函数中综合了不同的相似算子，以避免注意力偏差，并增强鲁棒性。通过与SC-CNN的组合，视图级别的全局交互也变得更加充分。通过一系列的真实数据实验与严谨的理论证明，该模型可以有效地建模复杂的用户风格特点等信息，并提升基于用户历史行为的风控/获客/经营/反欺诈等模型的性能。      
**论文链接**  [论文](https://dl.acm.org/doi/10.1145/3511808.3557284)
![deepvt图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/DEEPVT.PNG)      
<br/>    

### <span id='dml'>CIKM 2021: DML: Dynamic Multi-Granularity Learning for BERT-Based Document Reranking</span>    
**论文简介**  近年来，预训练的语言模型广泛应用于文本的检索排序任务中。然而，在真实场景中，用户的行为往往受到选择或曝光偏差的影响，这可能会导致错误的标签进而引入额外噪声。而对于不同候选文档，以往的训练优化目标通常使用单一粒度和静态权重。这使得排序模型的性能更容易受到上述问题的影响。因此，在本文中我们重点研究了基于BERT的文档重排序任务，开创性地提出了动态多粒度学习方法。该种方法能够让不同文档的权重根据预测概率动态变化，从而减弱不正确的文档标签带来的负面影响。此外，该方法还同时考虑了文档粒度和实例粒度来平衡候选文档的相对关系和绝对分数。在相关基准数据集上的实验进一步验证了我们模型的有效性。其中涉及到的预训练语言模型和排序模型为度小满获客、信贷、征信等业务提供了有效的文本特征和精准的排序结果，在促进业务稳健快速发展中起到了十分重要的作用。     
**论文链接**  [论文](https://dl.acm.org/doi/10.1145/3459637.3482090)     
<br/>    

### <span id='textvqa'>ACM MM 2021: Position-Augmented Transformers with Entity-Aligned Mesh for TextVQA</span>    
**论文简介**  许多图像除了实际的物体和背景等信息外，通常还包含着很有价值的文本信息，这对于理解图像场景是十分重要的。所以这篇文章主要研究了基于文本的视觉问答任务，这项任务要求机器可以理解图像场景并阅读图像中的文本来回答相应的问题。然而之前的大多数工作往往需要设计复杂的图结构和利用人工指定的特征来构建图像中视觉实体和文本之间的位置关系。而传统的多模态Transformer也不能有效地获取相对位置信息和原始图像特征。为了直观有效地解决这些问题，我们提出了一种新颖的模型，即具有实体对齐网格的位置增强Transformer。与之前的模型相比，我们在不需要复杂规则的情况下，显式地引入了目标检测和OCR识别的视觉实体的连续相对位置信息。此外，我们根据物体与OCR实体映射关系，用直观的实体对齐网格代替复杂的图形结构。在该网格中，不同位置的离散实体和图像的区块信息可以充分交互。相关实验显示，我们所提出的方法在多个基准数据集上超越了同类模型。而其中所用到目标检测、OCR以及基于Transformer的文本表示等方法源自度小满AI-Lab在CV和NLP领域的基础技术积累，已服务于信贷保险等多个业务场景。     
**论文链接**  [论文](https://dl.acm.org/doi/10.1145/3474085.3475425)   
![deepvt图片](https://github.com/Duxiaoman-DI/public-achievements-on-NLP/blob/main/ACM%20MM.png)     
<br/>    

### <span id='news'>WWW 2021: Combining Explicit Entity Graph with Implicit Text Information for News Recommendation</span>    
**论文简介**   精确地学习新闻和用户的表示是新闻推荐的核心问题。现有的模型通常侧重于利用隐含的文本信息来学习相应的表示，但这不足以建模用户兴趣。即使考虑了来自外部知识的实体信息，它也可能没有被明确和有效地用于用户建模。我们提出了一种新颖的新闻推荐方法，它将显式的实体图与隐含的文本信息相结合。实体图由两种类型的节点和三种类型的边组成，它们分别表示时间顺序、相关性和从属关系。然后利用图神经网络对这些节点进行推理。我们在一个真实的数据集——微软新闻数据集（MIND）上进行了实验，验证了我们提出的方法的有效性。       
**论文链接**  [论文](https://dl.acm.org/doi/10.1145/3442442.3452329)   
<br/>    


