# 中文计算机/AI文献检索题目集（computer 版）

用途：用于测试 `search_mode: computer` 下的中文计算机科学、人工智能与交叉技术主题到英文检索式生成、检索源选择、相关性筛选和后续文献分析流程。题目覆盖大模型、检索增强生成、自然语言处理、计算机视觉、机器学习、数据挖掘、软件工程、网络安全、数据库、分布式系统、人机交互和机器人算法等场景。

推荐默认设置：
- `search_mode`: `computer`
- 推荐检索源：arXiv、Semantic Scholar、OpenAlex、Crossref、CNKI；涉及医学 AI、公共卫生 AI、心理健康 AI 时可加 PubMed。
- 中文题目若选择 CNKI，应保留中文原始 query；英文源应使用英文技术术语和常用缩写。

标注位置：本文件只保存“检索题目”。每次实际检索后，候选文献会自动追加到 `检索标注记录.md`，请在那里填写 `人工判断`、`错误类型` 和 `备注`。

常用错误类型：
- `query_too_broad`：检索式泛化到大主题，如把 RAG 问答泛化为 LLM。
- `query_too_narrow`：检索式过窄，漏掉常见缩写或同义表达。
- `missing_synonym`：缺少 LLM、RAG、GNN、ViT、federated learning 等缩写/同义词。
- `wrong_source`：检索源不适合主题，如纯 CS 题目过度依赖 PubMed。
- `off_topic_passed`：跑题文献通过筛选。
- `relevant_rejected`：相关文献被误拒。
- `metadata_noise`：标题/摘要/元数据不足导致误判。
- `translation_error`：中文技术概念翻译错误。
- `concept_dropped`：任务、方法、数据集、指标、应用场景等核心概念丢失。
- `domain_mismatch`：应用领域不匹配，如法律问答返回医学问答。

## 题目清单

说明：每个题目是一个独立条目。字段保持一致，便于抽样测试和人工标注。

### C001 法律问答中的检索增强生成基准评测

- 领域：大语言模型 / 信息检索
- 核心概念：retrieval augmented generation；RAG；legal question answering；benchmark；evaluation
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：法律问答、法律检索、RAG 评测、法律领域 LLM benchmark 相关研究
- 应拒文献：通用 LLM 综述但无 RAG 或法律问答；医学问答 RAG；传统法律信息检索无生成模型
- 标注备注：防止泛化为 large language models 或 question answering

### C002 医学知识库问答中的大语言模型幻觉检测

- 领域：大语言模型 / 健康 AI
- 核心概念：large language model；hallucination detection；medical knowledge base；question answering；factuality
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；PubMed
- 应收文献：医学问答 LLM 幻觉检测、事实一致性评估、医学知识库 grounding、临床 QA 安全评估
- 应拒文献：非医学通用幻觉检测；纯医学知识图谱构建；医生问卷调查无 LLM
- 标注备注：此题可用 PubMed，但仍应走 computer/health-AI 策略

### C003 长文档摘要中的层次化 Transformer 模型

- 领域：自然语言处理
- 核心概念：long document summarization；hierarchical Transformer；abstractive summarization；document encoding
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：长文档摘要、层次化编码、Transformer 摘要模型、长上下文摘要评测
- 应拒文献：短文本摘要；纯长上下文语言模型无摘要任务；文档分类
- 标注备注：保留 long document 和 hierarchical 两个概念

### C004 中文医学文本命名实体识别的预训练语言模型

- 领域：自然语言处理 / 医学 AI
- 核心概念：Chinese medical named entity recognition；NER；pretrained language model；BERT；clinical text
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI；PubMed
- 应收文献：中文医学 NER、临床实体识别、BERT/预训练模型在医学文本上的应用
- 应拒文献：英文通用 NER；医学图像分割；中文情感分析
- 标注备注：CNKI 用中文原题，英文源用 Chinese medical NER

### C005 低资源机器翻译中的多语言预训练模型

- 领域：自然语言处理
- 核心概念：low-resource machine translation；multilingual pretrained model；cross-lingual transfer；mBART；mT5
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：低资源语言翻译、多语言预训练、跨语言迁移、神经机器翻译
- 应拒文献：高资源翻译任务；纯语言模型预训练无翻译；语音翻译
- 标注备注：不要丢掉 low-resource

### C006 代码生成模型的单元测试自动生成

- 领域：软件工程 / 大模型
- 核心概念：code generation model；unit test generation；LLM；program repair；software testing
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：LLM 代码生成、单元测试生成、测试用例生成、程序修复中的测试生成
- 应拒文献：人工测试管理；硬件测试；教育场景编程练习无自动测试生成
- 标注备注：unit test generation 是核心任务

### C007 软件缺陷预测中的图神经网络方法

- 领域：软件工程 / 图学习
- 核心概念：software defect prediction；graph neural network；GNN；code graph；bug prediction
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：软件缺陷预测、代码图表示、GNN bug prediction、静态代码图模型
- 应拒文献：通用图分类；缺陷检测但非软件；传统度量模型无 GNN
- 标注备注：区分 defect prediction 和 vulnerability detection

### C008 联邦学习中的差分隐私优化

- 领域：机器学习 / 隐私计算
- 核心概念：federated learning；differential privacy；privacy-preserving machine learning；optimization；communication efficiency
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：联邦学习差分隐私、隐私预算、DP-SGD、通信与精度权衡
- 应拒文献：集中式差分隐私；安全多方计算无联邦学习；联邦学习综述但无隐私
- 标注备注：federated learning 和 differential privacy 都必须保留

### C009 非独立同分布数据下的联邦学习聚合算法

- 领域：机器学习 / 联邦学习
- 核心概念：federated learning；non-IID data；aggregation algorithm；client drift；FedAvg
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：non-IID 联邦学习、聚合策略、客户端漂移、个性化/鲁棒聚合
- 应拒文献：集中式分布偏移；隐私保护但无 non-IID；边缘计算调度无联邦学习
- 标注备注：不要泛化为 generic distributed learning

### C010 图神经网络在交通流预测中的时空建模

- 领域：图学习 / 智能交通
- 核心概念：graph neural network；traffic flow prediction；spatio-temporal modeling；road network；time series forecasting
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：GNN/ST-GCN/时空图网络用于交通流、速度、拥堵预测
- 应拒文献：交通政策研究；普通时间序列预测无路网图；路径规划无流量预测
- 标注备注：保留交通预测应用场景

### C011 知识图谱补全中的关系路径推理模型

- 领域：知识图谱 / 表示学习
- 核心概念：knowledge graph completion；relation path reasoning；link prediction；path-based reasoning；embedding
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：KG completion、路径推理、关系路径、link prediction、可解释知识图谱推理
- 应拒文献：知识图谱构建无补全；普通图神经网络分类；信息抽取无推理
- 标注备注：区分 completion 和 construction

### C012 推荐系统中的多兴趣用户建模

- 领域：推荐系统
- 核心概念：recommender system；multi-interest user modeling；sequential recommendation；user representation；attention
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：多兴趣建模、序列推荐、用户表示、多向量召回、兴趣胶囊网络
- 应拒文献：单一兴趣协同过滤；广告点击率预测无多兴趣；社交网络用户画像
- 标注备注：multi-interest 是核心，不要泛化为 recommendation

### C013 生成式推荐系统中的大语言模型重排序

- 领域：推荐系统 / 大模型
- 核心概念：generative recommender system；large language model；reranking；LLM recommender；personalization
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：LLM 推荐、生成式推荐、候选重排序、个性化生成
- 应拒文献：传统推荐排序；通用 LLM 对话；生成图像推荐无文本/用户偏好
- 标注备注：reranking 和 recommender system 都要保留

### C014 图像分割中的 Segment Anything 模型迁移

- 领域：计算机视觉 / 基础模型
- 核心概念：Segment Anything Model；SAM；image segmentation；domain adaptation；transfer learning
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：SAM 在新领域图像分割、迁移学习、promptable segmentation、domain adaptation
- 应拒文献：传统 U-Net 分割无 SAM；通用 foundation model 综述；目标检测
- 标注备注：如果是医学图像 SAM 也可收，但标注领域差异

### C015 遥感图像变化检测中的孪生网络

- 领域：计算机视觉 / 遥感
- 核心概念：remote sensing image change detection；Siamese network；bi-temporal images；deep learning
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：遥感变化检测、双时相图像、孪生网络、建筑/土地覆盖变化检测
- 应拒文献：医学图像变化检测；遥感分类无变化检测；普通目标跟踪孪生网络
- 标注备注：remote sensing 和 change detection 必须保留

### C016 小样本目标检测中的元学习方法

- 领域：计算机视觉 / 小样本学习
- 核心概念：few-shot object detection；meta-learning；novel class detection；transfer learning
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：小样本目标检测、元学习、few-shot detection benchmark、novel class
- 应拒文献：小样本分类；零样本检测；通用目标检测无 few-shot
- 标注备注：区分 detection 和 classification

### C017 视频动作识别中的时空 Transformer

- 领域：计算机视觉 / 视频理解
- 核心概念：video action recognition；spatio-temporal Transformer；video understanding；attention
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：视频动作识别、时空注意力、视频 Transformer、行为识别 benchmark
- 应拒文献：静态图像分类；动作检测 localization-only；人体姿态估计无动作识别
- 标注备注：action recognition 是核心任务

### C018 点云语义分割中的 Transformer 网络

- 领域：三维视觉
- 核心概念：point cloud semantic segmentation；Transformer；3D vision；LiDAR；scene understanding
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：点云语义分割、3D Transformer、LiDAR 场景理解、室内/自动驾驶点云分割
- 应拒文献：2D 图像分割；点云配准；3D 目标检测无语义分割
- 标注备注：区分 semantic segmentation 和 object detection

### C019 自动驾驶感知中的多模态融合

- 领域：自动驾驶 / 多模态学习
- 核心概念：autonomous driving perception；multimodal fusion；camera；LiDAR；radar；BEV
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：自动驾驶感知融合、摄像头-LiDAR-radar、BEV 融合、目标检测/分割
- 应拒文献：自动驾驶政策；车辆控制无感知；单模态视觉检测
- 标注备注：不要漏掉 multimodal fusion

### C020 强化学习在机器人路径规划中的应用

- 领域：机器人 / 强化学习
- 核心概念：reinforcement learning；robot path planning；navigation；motion planning；obstacle avoidance
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：机器人导航、路径规划、强化学习避障、移动机器人/机械臂规划
- 应拒文献：传统 A* 综述无 RL；交通路径规划；游戏强化学习
- 标注备注：可接受 motion planning，但必须是机器人场景

### C021 多智能体强化学习中的信用分配问题

- 领域：强化学习 / 多智能体系统
- 核心概念：multi-agent reinforcement learning；credit assignment；cooperative MARL；value decomposition
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：MARL 信用分配、VDN/QMIX、合作多智能体、个体贡献估计
- 应拒文献：单智能体 RL；经济学信用评分；多机器人控制但无 RL
- 标注备注：credit assignment 容易被误解为金融信用

### C022 异常检测中的自监督表示学习

- 领域：机器学习 / 异常检测
- 核心概念：anomaly detection；self-supervised learning；representation learning；out-of-distribution detection
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：自监督异常检测、无监督异常检测、表示学习、工业/图像/时序异常
- 应拒文献：监督故障分类；OOD 分类但非异常检测；纯统计过程控制
- 标注备注：根据应用可标注图像、时序或工业场景

### C023 时间序列预测中的 Transformer 长期依赖建模

- 领域：时间序列 / 深度学习
- 核心概念：time series forecasting；Transformer；long-term dependency；long horizon forecasting；attention
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：长序列时间预测、Informer/Autoformer/FEDformer 类模型、长期依赖建模
- 应拒文献：NLP Transformer；短期预测传统 ARIMA；异常检测
- 标注备注：time series forecasting 和 long horizon 应保留

### C024 因果发现中的深度学习方法

- 领域：因果机器学习
- 核心概念：causal discovery；deep learning；DAG learning；structure learning；causal graph
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：深度因果发现、DAG 学习、连续优化因果结构学习、神经因果图
- 应拒文献：因果效应估计但无结构发现；社会科学因果识别；图神经网络普通链接预测
- 标注备注：区分 causal discovery 和 causal inference

### C025 公平机器学习中的偏差缓解算法

- 领域：可信 AI / 公平性
- 核心概念：fair machine learning；bias mitigation；algorithmic fairness；classification；fairness constraints
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：公平性约束、偏差缓解、pre/in/post-processing fairness、分类公平性指标
- 应拒文献：社会公平政策研究；数据隐私无公平；解释性 AI 无偏差缓解
- 标注备注：fairness 与 bias mitigation 都要保留

### C026 对抗样本防御中的鲁棒训练

- 领域：安全机器学习
- 核心概念：adversarial examples；adversarial defense；robust training；adversarial training；deep neural networks
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：对抗训练、鲁棒训练、防御方法、对抗样本检测与鲁棒性评估
- 应拒文献：网络入侵检测；模型压缩；普通图像增强
- 标注备注：不要泛化为 cybersecurity

### C027 深度伪造检测中的频域特征学习

- 领域：多媒体安全
- 核心概念：deepfake detection；frequency-domain features；forgery detection；face manipulation；deep learning
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：deepfake 检测、频域特征、伪造人脸检测、篡改检测
- 应拒文献：普通人脸识别；视频压缩；虚假新闻检测无视觉伪造
- 标注备注：frequency-domain 是关键限定

### C028 网络入侵检测中的深度学习模型

- 领域：网络安全
- 核心概念：network intrusion detection；deep learning；IDS；traffic classification；cybersecurity
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：网络流量入侵检测、深度学习 IDS、异常流量分类、攻击检测数据集
- 应拒文献：主机恶意软件检测；对抗样本安全；传统规则防火墙
- 标注备注：IDS 和 network traffic 要保留

### C029 恶意软件检测中的图表示学习

- 领域：网络安全 / 图学习
- 核心概念：malware detection；graph representation learning；control flow graph；call graph；GNN
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：恶意软件图表示、控制流图、调用图、GNN malware detection
- 应拒文献：网络钓鱼检测；普通软件缺陷预测；二进制分析无图学习
- 标注备注：malware 与 graph representation learning 都要保留

### C030 数据库查询优化中的学习型代价模型

- 领域：数据库 / 学习型系统
- 核心概念：database query optimization；learned cost model；cardinality estimation；query optimizer；machine learning
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：学习型查询优化、代价模型、基数估计、数据库优化器机器学习
- 应拒文献：通用机器学习优化；NoSQL 系统设计；索引结构无查询优化
- 标注备注：不要把 query 理解成搜索 query

### C031 分布式系统中的拜占庭容错共识算法

- 领域：分布式系统
- 核心概念：Byzantine fault tolerance；consensus algorithm；distributed systems；BFT；blockchain consensus
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：BFT 共识、拜占庭容错、分布式一致性、区块链共识协议
- 应拒文献：普通容灾备份；机器学习共识聚类；社会共识形成
- 标注备注：consensus 在这里是分布式一致性

### C032 边缘计算中的任务卸载与资源调度

- 领域：边缘计算 / 系统优化
- 核心概念：edge computing；task offloading；resource scheduling；mobile edge computing；latency
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：MEC 任务卸载、资源分配、延迟优化、边云协同调度
- 应拒文献：云计算综述无边缘；联邦学习边缘部署但无卸载；网络切片纯通信
- 标注备注：task offloading 是核心任务

### C033 可解释人工智能在图像分类中的显著性方法

- 领域：可解释 AI / 计算机视觉
- 核心概念：explainable AI；image classification；saliency methods；interpretability；CNN
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：图像分类解释、显著性图、Grad-CAM、CNN 可解释性
- 应拒文献：文本解释性；模型公平性；医学图像诊断但无解释方法
- 标注备注：saliency methods 是关键限定

### C034 人机交互中的可穿戴设备手势识别

- 领域：人机交互 / 可穿戴计算
- 核心概念：human-computer interaction；wearable device；gesture recognition；sensor data；interaction
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref；CNKI
- 应收文献：可穿戴传感器手势识别、HCI 交互、IMU/肌电手势、用户交互评估
- 应拒文献：视觉手势识别无可穿戴；医学康复动作识别；普通活动识别
- 标注备注：wearable device 与 gesture recognition 都要保留

### C035 隐私保护数据发布中的合成数据生成

- 领域：隐私计算 / 数据生成
- 核心概念：privacy-preserving data publishing；synthetic data generation；differential privacy；data utility；generative model
- 推荐检索源：arXiv；Semantic Scholar；OpenAlex；Crossref
- 应收文献：隐私保护合成数据、差分隐私生成模型、数据发布效用评估
- 应拒文献：普通数据增强；联邦学习无数据发布；隐私政策研究
- 标注备注：区分 synthetic data generation 和 data augmentation
