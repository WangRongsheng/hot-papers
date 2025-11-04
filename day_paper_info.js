const papersData = {
  "papers": [
    {
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们介绍了 Kimi Linear，这是一种混合线性注意力架构，首次在各种场景下（包括短上下文、长上下文和强化学习（RL）缩放机制）实现了在公平比较下超越全注意力的性能。其核心是 Kimi Delta Attention (KDA)，这是一个富有表现力的线性注意力模块，扩展了带闩锁的 DeltaNet，并具有更细粒度的门控机制，有效利用有限的有限状态 RNN 内存。我们定制的块状算法通过一种特殊的对角-加-低秩（DPLR）转移矩阵变体，实现了高硬件效率，大幅降低了计算量，相比于一般的 DPLR 公式，保持了更好的与经典增量规则的一致性。\n\n我们预训练了一个 Kimi Linear 模型，激活参数为 30 亿，总参数为 480 亿，基于 KDA 和多头潜在注意力（MLA）的层次混合。实验结果表明，在相同的训练方案下，Kimi Linear 在所有评估任务中都显著超越了全 MLA，同时将 KV 缓存使用减少了高达 75%，并在 1M 上下文下达到了最多 6 倍的解码吞吐量。这些结果表明，Kimi Linear 可以作为全注意力架构的替代品，提供更佳的性能和效率，包括输入和输出长度更长的任务。\n\n为了支持进一步的研究，我们已开源 KDA 内核和 vLLM 实现，并发布了预训练和调优后的模型检查点。",
      "paper_summary": {
        "summary": "Kimi Linear introduces a hybrid attention architecture combining a novel linear attention module with full attention layers, providing superior performance and efficiency for large language models. The architecture achieved up to 6x faster decoding throughput and 75% less KV cache usage compared to full attention baselines, while consistently matching or surpassing their quality across various tasks.",
        "originalProblem": [
          "Computational and memory bottlenecks of standard softmax attention in large language models (LLMs), which scale quadratically with sequence length.",
          "Limitations of existing pure linear attention mechanisms, often underperforming full attention in expressivity and long-context retrieval due to finite-state memory.",
          "High computational demands for 'agentic' LLMs that process extended trajectories and complex decision-making during inference."
        ],
        "solution": [
          "Develops Kimi Delta Attention (KDA), a linear attention module featuring fine-grained, channel-wise gating and a hardware-efficient chunkwise algorithm for precise memory control.",
          "Implements a hybrid architecture, Kimi Linear, by interleaving KDA layers with Multi-Head Latent Attention (MLA) layers in a uniform 3:1 ratio.",
          "Utilizes No Position Encoding (NoPE) in MLA layers, delegating all positional information and recency bias to the KDA layers to simplify MLA and remove complex RoPE adjustments."
        ],
        "keyInsights": [
          "Fine-grained, channel-wise gating in linear attention (KDA) significantly enhances expressivity and memory control, outperforming previous linear attention variants on synthetic tasks.",
          "A strategic hybrid architecture, combining efficient linear attention with selective full attention, can overcome the quality limitations of purely linear models while providing substantial efficiency gains.",
          "Delegating positional encoding responsibilities entirely to the linear attention component allows for simplified full attention layers (MLA with NoPE) and improves long-context robustness."
        ],
        "results": [
          "Achieved up to 6x faster decoding throughput and 75% reduced KV cache usage for 1M context lengths compared to full attention MLA baselines.",
          "Consistently outperformed full-attention MLA and hybrid Gated DeltaNet (GDN-H) baselines across general, reasoning, long-context, and RL benchmarks in fair, large-scale comparisons.",
          "Demonstrated approximately 1.16x computational efficiency during compute-optimal pre-training compared to MLA baselines."
        ]
      },
      "image_url": "image/2510.26692v2.png",
      "universal_paper_id": "2510.26692",
      "metrics": {
        "total_votes": 49,
        "visits_count": {
          "all": 1748,
          "last_7_days": 1748
        },
        "public_total_votes": 125
      },
      "first_publication_date": "2025-10-30T16:59:43.000Z",
      "publication_date": "2025-11-01T12:05:18.000Z",
      "updated_at": "2025-10-31T02:46:50.897Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.CL",
        "cs.LG",
        "efficient-transformers",
        "generative-models",
        "hardware-aware-algorithms",
        "instruction-tuning",
        "lightweight-models",
        "reinforcement-learning",
        "sequence-modeling",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 270,
      "github_url": "https://github.com/MoonshotAI/Kimi-Linear",
      "distance": 1
    },
    {
      "id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "paper_group_id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "title": "Context Engineering 2.0: The Context of Context Engineering",
      "abstract": "卡尔·马克思曾写道：“人类本质是社会关系的总和”，这表明个体并不是孤立的实体，而是根本上受到与其他实体的互动所塑造，其中情境扮演着构成性和基本的角色。随着计算机和人工智能的出现，这些情境不再仅限于纯粹的人际互动：人机互动也被纳入其中。那么，一个核心问题出现了：机器如何才能更好地理解我们的情境和目的？为了解决这一挑战，研究人员最近提出了情境工程的概念。尽管它通常被视为代理时代的最新创新，我们认为相关实践可以追溯到二十多年前。自1990年代初以来，这一领域经历了不同的历史阶段，每个阶段都受到机器智能水平的影响：从围绕原始计算机构建的早期人机互动框架，到如今由智能代理驱动的人-代理互动范式，未来可能走向人类水平或超人类智能。在本文中，我们定位了情境工程，提供了系统的定义，概述了其历史和概念背景，并审查了实践中的关键设计考虑因素。通过解决这些问题，我们旨在为情境工程提供概念基础，并勾勒其光明的未来。本文是更广泛社区在人工智能系统中进行系统性情境工程努力的一块踏脚石。",
      "paper_summary": null,
      "image_url": "image/2510.26493v1.png",
      "universal_paper_id": "2510.26493",
      "metrics": {
        "total_votes": 17,
        "visits_count": {
          "all": 740,
          "last_7_days": 740
        },
        "public_total_votes": 53
      },
      "first_publication_date": "2025-10-30T13:43:10.000Z",
      "publication_date": "2025-10-30T13:43:10.000Z",
      "updated_at": "2025-11-02T01:02:40.440Z",
      "topics": [
        "agent-based-systems",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "human-ai-interaction",
        "ml-systems",
        "reasoning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 119,
      "github_url": "https://github.com/jettbrains/-L-",
      "distance": 1
    },
    {
      "id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "paper_group_id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "title": "Continuous Autoregressive Language Models",
      "abstract": "大型语言模型（LLMs）的效率在根本上受限于它们逐 token 生成的顺序过程。我们认为，克服这一瓶颈需要为 LLM 扩展提供一个新的设计轴心：增加每个生成步骤的语义带宽。为此，我们引入了连续自回归语言模型（CALM），这是一种从离散下一个 token 预测转变为连续下一个向量预测的范式。CALM 使用高保真自编码器将一块 K 个 token 压缩成一个单一的连续向量，从中可以以超过 99.9% 的准确率重建原始 token。这使我们能够将语言建模为一系列连续向量，而不是离散 token，从而将生成步骤的数量降低一个 K 的因素。这一范式转变需要一个新的建模工具包；因此，我们开发了一个全面的无似然框架，使得在连续领域中能够进行稳健的训练、评估和可控采样。实验表明，CALM 显著改善了性能与计算的权衡，以显著较低的计算成本达到强离散基线的性能。更重要的是，这些发现确立了下一个向量预测作为通往超高效语言模型的强大且可扩展的途径。代码：此 https URL。项目：此 https URL。",
      "paper_summary": {
        "summary": "Continuous Autoregressive Language Models (CALM) replace discrete next-token prediction with continuous next-vector prediction to enhance efficiency in Large Language Models. A CALM-M model with 371M parameters achieved comparable performance to a 281M Transformer-S baseline, while reducing training FLOPs by 44% and inference FLOPs by 34%.",
        "originalProblem": [
          "Traditional Large Language Models (LLMs) face inherent efficiency bottlenecks due to their token-by-token autoregressive generation, which is computationally expensive for long sequences.",
          "Discrete subword tokens carry limited semantic information, requiring numerous sequential steps to generate meaningful text and leading to high computational demands.",
          "The approach of increasing vocabulary size to pack more information into discrete tokens is computationally intractable, limiting further efficiency gains."
        ],
        "solution": [
          "CALM introduces a paradigm shift from discrete next-token prediction to continuous next-vector prediction for language generation.",
          "A robust, high-fidelity autoencoder is developed to compress chunks of K discrete tokens into a single, dense continuous latent vector, achieving over 99.9% reconstruction accuracy.",
          "A comprehensive likelihood-free framework supports the continuous models, featuring an Energy Transformer for single-step continuous vector generation, BrierLM for evaluation, and a likelihood-free algorithm for temperature sampling."
        ],
        "keyInsights": [
          "Increasing the semantic bandwidth of each generative step, by predicting continuous vectors representing multiple tokens, provides a new pathway for scaling LLM efficiency.",
          "Robust and smooth continuous latent representations are critical for generative modeling, requiring variational regularization, KL clipping, and dropout in the autoencoder design.",
          "Likelihood-free objectives and evaluation metrics are necessary and effective for training and assessing continuous generative language models without explicit probability distributions."
        ],
        "results": [
          "CALM models demonstrated a superior performance-compute trade-off, with a 371M parameter CALM-M reducing training FLOPs by 44% and inference FLOPs by 34% compared to a 281M Transformer-S baseline at comparable performance.",
          "Varying the semantic bandwidth (chunk size K) revealed it as a critical lever for optimizing efficiency, with K=4 models surpassing the performance-compute frontier of standard Transformers.",
          "The Energy Transformer generative head achieved superior single-step continuous generation quality compared to diffusion and flow matching heads, eliminating iterative inference bottlenecks."
        ]
      },
      "image_url": "image/2510.27688v1.png",
      "universal_paper_id": "2510.27688",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 272,
          "last_7_days": 272
        },
        "public_total_votes": 25
      },
      "first_publication_date": "2025-10-31T17:58:11.000Z",
      "publication_date": "2025-10-31T17:58:11.000Z",
      "updated_at": "2025-11-03T03:02:51.158Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "efficient-transformers",
        "generative-models",
        "model-compression",
        "representation-learning",
        "sequence-modeling",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 6,
      "github_url": "https://github.com/shaochenze/calm",
      "distance": 1
    },
    {
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们介绍了Emu3.5，这是一个大规模的多模态世界模型，能够原生地预测视觉和语言之间的下一个状态。Emu3.5在一个包含超过10万亿个标记的视觉语言交织数据语料库上进行了端到端的预训练，主要来源于互联网视频的连续帧和文本记录。该模型自然接受交织的视觉语言输入并生成交织的视觉语言输出。Emu3.5还通过大规模强化学习进行后训练，以增强多模态推理和生成。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐个标记解码转换为双向并行预测，使每幅图像的推理速度提高约20倍，而不影响性能。Emu3.5展现了强大的原生多模态能力，包括长时限视觉语言生成、任意到图像（X2I）生成和复杂的文本丰富图像生成。它还展现了可普遍应用的世界建模能力，能够在多种场景和任务中实现时空一致的世界探索和开放世界的具身操作。作为比较，Emu3.5在图像生成和编辑任务上的表现可与Gemini 2.5 Flash Image（Nano Banana）相媲美，并在一系列交织生成任务中展示了更优的结果。我们将在此网址开源Emu3.5，以支持社区研究。",
      "paper_summary": {
        "summary": "BAAI's Emu3.5 introduces a native multimodal model trained extensively on internet videos, capable of understanding and generating interleaved vision-language sequences across long horizons. The model achieves strong performance in various multimodal tasks, including complex image generation with accurate text rendering and embodied AI scenarios, while accelerating image inference by approximately 20x using a novel Discrete Diffusion Adaptation (DiDA) technique.",
        "originalProblem": [
          "Large language models are inherently limited by their text-only input, offering a restricted view of the world and hindering comprehensive understanding.",
          "Many existing vision-language models employ separate encoders or adapter-based approaches, lacking deep, native integration of vision and language for unified prediction.",
          "Autoregressive models suffer from slow, token-by-token inference, creating a practical bottleneck for generating high-resolution images in real-time applications."
        ],
        "solution": [
          "Develop Emu3.5, a 34.1 billion parameter decoder-only transformer, designed for native multimodal prediction of the 'next state across vision and language' using a unified next-token prediction objective.",
          "Pre-train the model on over 13 trillion multimodal tokens, primarily sourced from ~63 million internet videos, to learn long-horizon spatiotemporal continuity and cross-modal alignment.",
          "Introduce Discrete Diffusion Adaptation (DiDA), a technique that converts autoregressive image generation into bidirectional parallel prediction, significantly accelerating inference.",
          "Apply extensive post-training through two-stage supervised fine-tuning (SFT) and large-scale reinforcement learning (RL) guided by comprehensive multimodal rewards to enhance reasoning and generation."
        ],
        "keyInsights": [
          "Native multimodal architectures, when trained on vast quantities of long-horizon interleaved vision-language data, move closer to learning a 'world model' that captures temporal dynamics and rich context.",
          "The Discrete Diffusion Adaptation (DiDA) method effectively transforms sequential autoregressive image generation into an efficient parallel process, addressing a major practical limitation for high-resolution output.",
          "A unified post-training strategy, incorporating large-scale SFT and RL with a comprehensive multimodal reward system, is crucial for aligning model outputs with human preferences and improving generalization across diverse tasks.",
          "Scaling pre-training compute with high-quality, meticulously filtered and annotated internet video data leads to progressively stronger generalization capabilities across multimodal tasks."
        ],
        "results": [
          "Emu3.5 achieves strong performance in Any-to-Image (X2I) generation, including open-world image editing, precise control, and complex text-rich image rendering, outperforming various state-of-the-art T2I models on benchmarks like LeX-Bench and CVTG-2K.",
          "The model demonstrates generalizable world-modeling abilities, achieving a 65.5% win rate against Gemini 2.5 Flash Image for world exploration and a 67.1% win rate for embodied manipulation in automated preference evaluations.",
          "Discrete Diffusion Adaptation (DiDA) accelerates per-image inference by approximately 20x (reducing generation time for a 1024x1024 image from 120s to 10s) without sacrificing performance on T2I and X2I tasks, making it competitive with fast-sampling diffusion models."
        ]
      },
      "image_url": "image/2510.26583v1.png",
      "universal_paper_id": "2510.26583",
      "metrics": {
        "total_votes": 21,
        "visits_count": {
          "all": 632,
          "last_7_days": 632
        },
        "public_total_votes": 70
      },
      "first_publication_date": "2025-10-30T15:11:16.000Z",
      "publication_date": "2025-10-30T15:11:16.000Z",
      "updated_at": "2025-10-31T02:17:25.338Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "multi-modal-learning",
        "reinforcement-learning",
        "representation-learning",
        "robotic-control",
        "transformers",
        "video-understanding",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "BAAI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 555,
      "github_url": "https://github.com/baaivision/Emu3.5",
      "distance": 1
    },
    {
      "id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "paper_group_id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "title": "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning",
      "abstract": "大型语言模型（LLMs）通常在需要多步推理的问题上表现不佳。对于小规模的开源模型，具有可验证奖励的强化学习（RLVR）在经过多次尝试后仍然难以抽取到正确解决方案，而监督微调（SFT）则往往通过僵化的逐字模仿导致对长示范的过拟合。为了解决这一问题，我们提出了监督强化学习（SRL），一个将问题解决重新定义为生成一系列逻辑“行动”的框架。SRL训练模型在每次行动之前生成内部推理独白。它基于模型的行动与从SFT数据集中提取的专家行动之间的相似性，逐步提供更平滑的奖励。这种监督即使在所有回合均错误的情况下也提供了更丰富的学习信号，同时鼓励灵活的推理，以专家示范为指导。因此，SRL使小模型能够学习以前无法通过SFT或RLVR学习的挑战性问题。此外，在用RLVR精细化之前以SRL初始化训练可获得最佳整体性能。超越推理基准，SRL还有效地泛化到代理软件工程任务，确立了它作为面向推理的大型语言模型的稳健且多功能的训练框架。",
      "paper_summary": {
        "summary": "The Supervised Reinforcement Learning (SRL) framework enables smaller Large Language Models (LLMs) to learn complex multi-step reasoning by providing dense, step-wise similarity rewards from expert demonstrations. It notably improved greedy average accuracy on math reasoning benchmarks by 3.0% over the base model, and showed a 74% relative improvement in resolve rate on agentic software engineering tasks.",
        "originalProblem": [
          "Small-scale, open-source LLMs struggle to solve challenging multi-step reasoning problems effectively with current training paradigms.",
          "Supervised Fine-Tuning (SFT) often leads to overfitting on complex expert demonstrations, sometimes causing performance degradation for smaller models.",
          "Reinforcement Learning with Verifiable Rewards (RLVR) faces sparse reward signals on difficult problems, hindering effective learning due to a lack of positive advantage estimates."
        ],
        "solution": [
          "Introduced Supervised Reinforcement Learning (SRL), which reformulates problem-solving as a sequential decision-making process by decomposing expert solutions into logical 'actions'.",
          "Generated step-wise training data from expert trajectories and provided a dense sequence similarity reward based on the model's predicted action, allowing for flexible internal monologues.",
          "Optimized the policy using a Group Relative Policy Optimization (GRPO) objective with dynamic sampling and explored a curriculum learning strategy combining SRL with RLVR."
        ],
        "keyInsights": [
          "Dense, step-wise supervision, through sequence similarity rewards on discrete actions, effectively guides LLMs on complex problems where sparse rewards or rigid imitation fail.",
          "SRL encourages flexible and sophisticated reasoning patterns, including planning, on-the-fly adjustments, and reflective verification, without merely increasing output length.",
          "The combination of SRL for initial robust guidance and subsequent RLVR refinement forms a powerful curriculum learning approach for maximizing performance on challenging tasks."
        ],
        "results": [
          "SRL alone achieved a 27.6% greedy average accuracy on math reasoning benchmarks, a 3.0% increase over the base Qwen2.5-7B-Instruct model.",
          "The SRL → RLVR pipeline delivered the strongest performance, reaching 28.3% greedy average accuracy, a 3.7% increase over the base model and outperforming all baselines.",
          "SRL generalized to agentic software engineering tasks, achieving a 14.8% resolve rate in an oracle file editing setting, representing a 74% relative improvement over an SFT baseline."
        ]
      },
      "image_url": "image/2510.25992v1.png",
      "universal_paper_id": "2510.25992",
      "metrics": {
        "total_votes": 12,
        "visits_count": {
          "all": 623,
          "last_7_days": 623
        },
        "public_total_votes": 60
      },
      "first_publication_date": "2025-10-29T22:05:08.000Z",
      "publication_date": "2025-10-29T22:05:08.000Z",
      "updated_at": "2025-10-31T02:17:59.140Z",
      "topics": [
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "fine-tuning",
        "imitation-learning",
        "reasoning",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "UCLA",
          "image": "images/organizations/ucla.png"
        },
        {
          "name": "Google Cloud",
          "image": null
        },
        {
          "name": "Google Cloud AI Research",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4816-97ae-711f-8c4a-c6055c39d3d6",
      "paper_group_id": "019a4816-97ae-711f-8c4a-c6055c39d3d6",
      "title": "Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning",
      "abstract": "空间理解仍然是大型视觉语言模型（LVLMs）的一个弱点。现有的监督微调（SFT）和最近的可验证奖励强化学习（RLVR）管道依赖于昂贵的监督、专业工具或受限环境，这限制了规模。我们引入了Spatial-SSRL，这是一种自监督的强化学习范式，直接从普通的RGB或RGB-D图像中获取可验证信号。Spatial-SSRL自动设计了五个前置任务，以捕捉2D和3D空间结构：打乱的补丁重排序、翻转的补丁识别、裁剪的补丁修复、区域深度排序和相对3D位置预测。这些任务提供了易于验证的真实答案，无需人工或LVLM注释。在我们的任务上进行训练显著提高了空间推理能力，同时保持一般视觉能力。在七个图像和视频设置的空间理解基准测试中，Spatial-SSRL在Qwen2.5-VL基线的基础上实现了平均准确率提升分别为4.63%（3B）和3.89%（7B）。我们的结果表明，简单的内在监督使得大规模的RLVR成为可能，并为LVLMs提供了更强的空间智能的实际途径。",
      "paper_summary": {
        "summary": "Researchers at Shanghai AI Laboratory developed Spatial-SSRL, a self-supervised reinforcement learning framework that enhances the spatial understanding of Large Vision-Language Models (LVLMs) by generating verifiable ground-truth signals from ordinary images. The method consistently improved performance by an average of +4.63% on diverse spatial benchmarks for the 3B model and preserved or improved general visual capabilities.",
        "originalProblem": [
          "Large Vision-Language Models (LVLMs) demonstrate weak spatial understanding, struggling with depth, distance, orientation, and relative object positions in 3D.",
          "Supervised Fine-Tuning (SFT) methods for spatial understanding are costly due to annotation needs and often lead to dataset-specific memorization rather than generalization.",
          "Reinforcement Learning with Verifiable Rewards (RLVR) approaches are limited by their dependence on specialized tools, synthetic environments, or specific public datasets, hindering scalability."
        ],
        "solution": [
          "A self-supervised reinforcement learning framework, Spatial-SSRL, is introduced to enhance LVLM spatial understanding.",
          "Five pretext tasks (three depth-free and two depth-based) are designed to generate deterministic, verifiable ground-truth answers from RGB and RGB-D images.",
          "Large Vision-Language Models are fine-tuned using Group Relative Policy Optimization (GRPO), leveraging a reward function based on answer accuracy and format adherence."
        ],
        "keyInsights": [
          "Intrinsic structural information within ordinary 2D (RGB) and 3D (RGB-D) images can serve as scalable, naturally verifiable ground truth for spatial reasoning tasks.",
          "Repurposing self-supervised learning objectives as reward functions within an RL framework effectively allows for cost-effective and scalable spatial intelligence acquisition.",
          "Combining diverse self-supervised tasks (depth-free and depth-based) synergistically improves overall spatial reasoning across various dimensions."
        ],
        "results": [
          "Spatial-SSRL consistently improved spatial understanding across seven benchmarks, with the 3B model achieving an average accuracy gain of +4.63% and the 7B model +3.89%.",
          "Models trained with Spatial-SSRL demonstrated enhanced reasoning capabilities, outperforming baselines when prompted for explicit reasoning chains.",
          "General visual capabilities, including VQA and fine-grained perception, were preserved and often improved, with the 3B model showing an average gain of +2.02% across general VQA benchmarks."
        ]
      },
      "image_url": "image/2510.27606v1.png",
      "universal_paper_id": "2510.27606",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 89,
          "last_7_days": 89
        },
        "public_total_votes": 14
      },
      "first_publication_date": "2025-10-31T16:30:08.000Z",
      "publication_date": "2025-10-31T16:30:08.000Z",
      "updated_at": "2025-11-03T05:00:31.534Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "deep-reinforcement-learning",
        "geometric-deep-learning",
        "multi-modal-learning",
        "representation-learning",
        "self-supervised-learning",
        "vision-language-models"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 14,
      "github_url": "https://github.com/InternLM/Spatial-SSRL",
      "distance": 1
    },
    {
      "id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "paper_group_id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "title": "Defeating the Training-Inference Mismatch via FP16",
      "abstract": "强化学习（RL）对大型语言模型（LLM）进行微调时，常常因训练策略与推理策略之间的数值不匹配而导致不稳定。尽管先前的工作试图通过算法修正或工程对齐来缓解这一问题，但我们指出其根本原因在于浮点精度本身。广泛采用的BF16尽管具有较大的动态范围，但引入了较大的舍入误差，破坏了训练与推理之间的一致性。在本研究中，我们展示了简单地恢复使用\\textbf{FP16}可以有效消除这种不匹配。这个改变简单，现代框架完全支持，只需少量代码更改，并且无需修改模型架构或学习算法。我们的结果表明，使用FP16可以均匀地提高优化的稳定性，加快收敛速度，并在不同任务、算法和框架中提升性能。我们希望这些发现能激励对RL微调中的精度权衡进行更广泛的重新考虑。",
      "paper_summary": {
        "summary": "This research from Sea AI Lab and the National University of Singapore demonstrates that using FP16 precision for both training and inference fundamentally resolves the numerical discrepancies causing instability in Reinforcement Learning (RL) fine-tuning of Large Language Models. Employing FP16 drastically reduces the training-inference mismatch, leading to more stable optimization, faster convergence, and superior performance compared to existing BF16-based methods and complex algorithmic corrections.",
        "originalProblem": [
          "Reinforcement Learning (RL) fine-tuning of Large Language Models (LLMs) often suffers from instability, training collapse, and sub-optimal performance.",
          "The 'training-inference mismatch,' caused by numerical discrepancies between fast inference and gradient computation engines, leads to biased gradients and a 'deployment gap'.",
          "Existing solutions, such as importance sampling corrections or manual engineering alignments, are computationally expensive, complex, or only address symptoms without eliminating the fundamental mismatch."
        ],
        "solution": [
          "The paper hypothesizes and confirms that low floating-point precision, specifically BF16, is the root cause of the training-inference mismatch due to accumulated rounding errors.",
          "The core solution involves switching from BF16 to FP16 precision for both the inference policy and the training engine in RL fine-tuning.",
          "FP16's higher numerical precision (more mantissa bits) ensures that minor implementation differences are absorbed, preventing policy divergence, with standard loss scaling addressing its limited dynamic range."
        ],
        "keyInsights": [
          "BF16's lower precision (7 mantissa bits) is a primary source of numerical instability in RL fine-tuning, causing policies to diverge even with identical model weights.",
          "FP16's significantly higher precision (10 mantissa bits) effectively eliminates the training-inference mismatch by ensuring numerical consistency between inference and training policies.",
          "By addressing the root cause of the mismatch, FP16 simplifies RL fine-tuning, allowing even simple, unbiased policy gradient estimators to perform robustly and effectively, rendering complex algorithmic corrections less necessary."
        ],
        "results": [
          "FP16 reduced the training-inference mismatch in sequence-level log-probability ratios by approximately 24 times compared to BF16.",
          "Switching to FP16 precision enabled traditional RL algorithms (e.g., GRPO, PG-Seq-IS) to converge stably and achieve nearly 100% training accuracy on a challenging dataset, while BF16-based methods frequently collapsed or converged slowly.",
          "The benefits of FP16 generalized across diverse settings, including Mixture-of-Experts (MoE) models, LoRA-based fine-tuning, larger dense models, and different LLM architectures, consistently leading to higher training rewards and validation performance."
        ]
      },
      "image_url": "image/2510.26788v1.png",
      "universal_paper_id": "2510.26788",
      "metrics": {
        "total_votes": 27,
        "visits_count": {
          "all": 671,
          "last_7_days": 671
        },
        "public_total_votes": 83
      },
      "first_publication_date": "2025-10-30T17:58:11.000Z",
      "publication_date": "2025-10-30T17:58:11.000Z",
      "updated_at": "2025-10-31T05:21:01.522Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "deep-reinforcement-learning",
        "fine-tuning",
        "hardware-aware-algorithms",
        "ml-systems",
        "optimization-methods",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 5,
      "github_url": "https://github.com/sail-sg/Precision-RL",
      "distance": 1
    },
    {
      "id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "paper_group_id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "title": "Scaling Latent Reasoning via Looped Language Models",
      "abstract": "现代的大型语言模型（LLM）主要通过显式文本生成进行“思考”，例如思维链（CoT），这使得推理被推迟到训练后，并且未充分利用预训练数据。我们提出并开源了Ouro，命名源于递归的乌洛波洛斯，它是一系列预训练的循环语言模型（LoopLM），通过(i) 潜在空间中的迭代计算，(ii) 用于学习深度分配的熵正则化目标，以及(iii) 扩展到7.7万亿个标记，在预训练阶段将推理构建进模型中。Ouro 1.4B和2.6B模型在广泛的基准测试中展现出卓越的性能，达到了高达120亿参数的最新技术水平（SOTA）LLM的结果。通过受控实验，我们表明，这一优势并非源于知识容量的增加，而是出于更好的知识操作能力。我们还展示了LoopLM能够生成与最终输出更一致的推理痕迹，而不是显式的CoT。我们希望我们的结果展示出LoopLM作为推理时代一种新型扩展方向的潜力。我们的模型可以在此链接找到。",
      "paper_summary": {
        "summary": "Researchers introduce Ouro, a family of Looped Language Models (LoopLM) that offers a new scaling pathway for large language models, achieving 2-3x parameter efficiency and improved reasoning capabilities by leveraging iterative latent computation. This approach demonstrates enhanced safety and more faithful reasoning, particularly on complex tasks.",
        "originalProblem": [
          "Traditional scaling of Large Language Models (LLMs) faces diminishing returns, leading to prohibitively high computational costs and data scarcity.",
          "Deploying massive LLMs is hindered by significant latency, cost, and infrastructure requirements, limiting accessibility.",
          "Current reasoning methods, such as Chain-of-Thought, extend output sequences, increasing inference cost and often provide post-hoc rationalizations rather than integrated latent reasoning."
        ],
        "solution": [
          "Developed Ouro, a family of Looped Language Models (LoopLM) that applies shared-weight transformer layers recurrently to scale computational depth without increasing parameter count.",
          "Integrated an adaptive computation mechanism that employs learned early exit gates, enabling dynamic allocation of recurrent steps based on input complexity.",
          "Trained these models using a multi-stage pre-training pipeline on 7.7 trillion tokens, incorporating an entropy-regularized loss and a specialized gate training objective for efficient learning."
        ],
        "keyInsights": [
          "LoopLMs achieve performance gains by enhancing knowledge manipulation and compositional reasoning capabilities, rather than increasing raw knowledge storage capacity.",
          "Iterative latent reasoning processes lead to a more causally faithful reasoning trace and a continuous improvement in safety alignment as recurrent steps increase.",
          "Recurrent depth represents a critical third scaling axis for LLMs, allowing for significant parameter efficiency by decoupling computational depth from the number of unique parameters."
        ],
        "results": [
          "Ouro 1.4B and 2.6B models demonstrate a 2-3x parameter efficiency, consistently matching or exceeding the performance of 4B and 8B parameter dense models, respectively, on challenging reasoning benchmarks like BBH, GSM8K, and MATH500.",
          "Adaptive computation mechanisms effectively allocate resources, with specialized 'Ponder gate' training achieving the best accuracy-efficiency trade-off and KV cache sharing reducing decoding memory requirements by 4x.",
          "Mechanistic studies confirmed LoopLMs excel at knowledge manipulation (e.g., higher sample efficiency on multi-hop QA) rather than increased knowledge capacity, and exhibit improved safety alignment with deeper recurrent steps."
        ]
      },
      "image_url": "image/2510.25741v1.png",
      "universal_paper_id": "2510.25741",
      "metrics": {
        "total_votes": 25,
        "visits_count": {
          "all": 1019,
          "last_7_days": 1019
        },
        "public_total_votes": 93
      },
      "first_publication_date": "2025-10-29T17:45:42.000Z",
      "publication_date": "2025-10-29T17:45:42.000Z",
      "updated_at": "2025-10-30T02:56:13.105Z",
      "topics": [
        "chain-of-thought",
        "Computer Science",
        "cs.CL",
        "lightweight-models",
        "mechanistic-interpretability",
        "optimization-methods",
        "parameter-efficient-training",
        "reasoning",
        "representation-learning",
        "self-supervised-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Carnegie Mellon University",
          "image": "images/organizations/cmu.jpg"
        },
        {
          "name": "University of Manchester",
          "image": "images/organizations/university-of-manchester.png"
        },
        {
          "name": "Mila - Quebec AI Institute",
          "image": "images/organizations/mila.jpeg"
        },
        {
          "name": "ByteDance",
          "image": "images/organizations/bytedance.png"
        },
        {
          "name": "Peking University",
          "image": "images/organizations/peking.png"
        },
        {
          "name": "University of Pennsylvania",
          "image": "images/organizations/upenn.jpeg"
        },
        {
          "name": "Princeton University",
          "image": "images/organizations/princeton.jpg"
        },
        {
          "name": "University of Montreal",
          "image": null
        },
        {
          "name": "University of California, Santa Cruz",
          "image": "images/organizations/ucsc.png"
        },
        {
          "name": "Conscium",
          "image": null
        },
        {
          "name": "M-A-P",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "paper_group_id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "title": "The Era of Agentic Organization: Learning to Organize with Language Models",
      "abstract": "我们展望一个新的人工智能时代，称为代理组织（agentic organization），在这个时代，代理能够通过协作和同时工作解决复杂问题，实现超越个人智能的成果。为了实现这一愿景，我们引入了异步思维（AsyncThink）作为与大型语言模型进行推理的新范式，它将内部思维过程组织成可以同时执行的结构。具体而言，我们提出了一种思维协议，其中组织者动态分配子查询给工作者，合并中间知识，生成连贯的解决方案。更重要的是，该协议中的思维结构可以通过强化学习进一步优化。实验表明，与并行思维相比，AsyncThink的推理延迟降低了28%，同时提高了数学推理的准确性。此外，AsyncThink对其学习到的异步思维能力进行了泛化，能够有效应对未见过的任务而无需额外训练。",
      "paper_summary": {
        "summary": "A new reasoning paradigm, AsyncThink, enables large language models to learn and execute self-organized, asynchronous thought processes. This approach achieves superior accuracy and reduces critical-path latency by 28% on mathematical reasoning tasks, and demonstrates strong generalization across various complex problem-solving scenarios.",
        "originalProblem": [
          "Current large language models (LLMs) face limitations in solving complex, multi-faceted problems that require more than sequential or simple parallel thinking.",
          "Existing parallel reasoning approaches suffer from high critical-path latency, as they must wait for the slowest independent reasoning trace to complete before aggregation.",
          "Manually designed, static reasoning workflows lack the adaptivity needed for diverse queries, hindering dynamic problem decomposition and execution."
        ],
        "solution": [
          "AsyncThink proposes an Organizer-Worker Thinking Protocol where a single LLM backbone dynamically manages its thought process, using \"Fork\" to delegate sub-queries and \"Join\" to integrate results.",
          "A two-stage training procedure starts with supervised fine-tuning on GPT-4o synthesized data to teach the protocol's text-based syntax.",
          "Reinforcement learning, utilizing a rule-based reward system, optimizes the model's ability to maximize correctness and thinking concurrency, adapting the Group Relative Policy Optimization (GRPO) algorithm for non-sequential traces."
        ],
        "keyInsights": [
          "Large language models can learn to dynamically self-organize their internal reasoning processes through \"Fork\" and \"Join\" operations, moving beyond static, predefined thinking workflows.",
          "Optimizing for both correctness and a quantifiable thinking concurrency reward during reinforcement learning is essential for achieving efficient and high-performing asynchronous problem-solving.",
          "The learned asynchronous thinking policies demonstrate strong generalization capabilities, allowing models to apply complex decomposition strategies to entirely unseen problem domains."
        ],
        "results": [
          "AsyncThink significantly increased \"All Correct\" solutions on Multi-Solution Countdown to 89.0%, surpassing parallel thinking (68.6%) and sequential thinking (70.5%).",
          "On mathematical reasoning benchmarks, it achieved comparable or superior accuracy while reducing critical-path latency by up to 28% compared to existing parallel thinking methods.",
          "The system demonstrated strong zero-shot generalization, achieving 89.4% accuracy on 4x4 Sudoku with lower latency, even when trained on different tasks."
        ]
      },
      "image_url": "image/2510.26658v1.png",
      "universal_paper_id": "2510.26658",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 355,
          "last_7_days": 355
        },
        "public_total_votes": 36
      },
      "first_publication_date": "2025-10-30T16:25:10.000Z",
      "publication_date": "2025-10-30T16:25:10.000Z",
      "updated_at": "2025-10-31T03:06:14.881Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4842-a558-7fc0-828b-44748ee828ed",
      "paper_group_id": "019a4842-a558-7fc0-828b-44748ee828ed",
      "title": "Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model",
      "abstract": "最近，通过世界建模增强视觉-语言-动作模型（VLA）在改善机器人策略学习方面展现了潜力。然而，由于这两种模态之间的固有差异，联合预测下一个状态观测和动作序列仍然具有挑战性。为了解决这个问题，我们提出了双流扩散框架（DUST），这是一个增强了世界模型的VLA框架，旨在处理模态冲突并提升VLA在各种任务中的性能。具体而言，我们提出了一种多模态扩散变换器架构，它明确保持独立的模态流，同时仍然允许跨模态知识共享。此外，我们为每种模态引入了独立的噪声扰动和解耦的流匹配损失。这个设计使得模型能够以双向的方式学习联合分布，而无需统一的潜在空间。基于训练过程中模态的解耦，我们还引入了一种联合采样方法，支持测试时的比例缩放，在该方法中，动作和视觉标记以不同的速率异步演变。通过在RoboCasa和GR-1等模拟基准上的实验，DUST相较于基线方法实现了高达6%的提升，而我们的测试时缩放方法提供了额外的2-5%增益。在使用Franka Research 3进行的真实世界任务中，DUST将成功率提高了13%，进一步确认了其在模拟之外的有效性。此外，在BridgeV2上对无动作视频进行预训练在RoboCasa上带来了显著的迁移增益，强调了DUST在大规模VLA预训练中的潜力。",
      "paper_summary": {
        "summary": "DUAL-STream diffusion (DUST), developed at KAIST, is a Vision-Language-Action (VLA) model that employs a dual-stream diffusion architecture to jointly predict future observations and actions, explicitly addressing modality conflicts for robotic control. The model achieves state-of-the-art performance across simulated and real-world tasks, showing a 13% average success rate improvement on real-world tasks and enhanced data efficiency through action-free video pre-training.",
        "originalProblem": [
          "Existing VLA models struggle with modality conflict when jointly predicting actions (low-dimensional) and future visual observations (high-dimensional), as unified approaches mix distinct data types and causal models limit bidirectional information flow.",
          "Robot policy learning often lacks an explicit understanding of physical processes, hindering generalization and the ability to anticipate how actions affect the environment.",
          "The high cost and labor-intensive nature of collecting expert robot demonstrations create a significant data scarcity bottleneck for training robust VLA models."
        ],
        "solution": [
          "A dual-stream multimodal diffusion transformer (MMDiT) architecture processes action and vision tokens in separate pathways, only temporarily concatenating them during a shared cross-modal attention layer for rich, bidirectional information exchange.",
          "A decoupled training algorithm applies independent noise levels and separate timesteps to action and future observation embeddings, enabling efficient joint optimization and learning of bidirectional causal dependencies.",
          "An asynchronous joint sampling strategy during inference allows vision tokens to undergo more denoising steps than action tokens, optimizing the trade-off between inference speed and visual prediction accuracy."
        ],
        "keyInsights": [
          "Maintaining separate processing streams for actions and vision while allowing deep, bidirectional cross-modal interaction effectively resolves modality conflicts, leading to improved VLA model performance and learning efficiency.",
          "Decoupled noise scheduling during training is crucial for efficiently optimizing both action prediction and world modeling objectives and learning their bidirectional causal relationships.",
          "Leveraging large-scale, action-free video data for pre-training significantly enhances data efficiency and generalization capabilities for downstream robotic policy learning tasks."
        ],
        "results": [
          "DUST improved average success rates by 18% over GR00T-N1.5 and 5% over FLARE on RoboCasa simulated kitchen tasks with 100 demonstrations, demonstrating strong data efficiency.",
          "The model achieved a 13% average success rate improvement over GR00T-N1.5 and 12% over FLARE on real-world Franka Research 3 pick-and-place tasks.",
          "Pre-training DUST on action-free video data from BridgeV2 significantly enhanced transfer learning, boosting the average success rate on RoboCasa from 0.501 to 0.585 with limited robot demonstrations."
        ]
      },
      "image_url": "image/2510.27607v1.png",
      "universal_paper_id": "2510.27607",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 50,
          "last_7_days": 50
        },
        "public_total_votes": 11
      },
      "first_publication_date": "2025-10-31T16:32:12.000Z",
      "publication_date": "2025-10-31T16:32:12.000Z",
      "updated_at": "2025-11-03T05:48:38.616Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.RO",
        "generative-models",
        "inference-optimization",
        "multi-modal-learning",
        "reinforcement-learning",
        "robotic-control",
        "robotics-perception",
        "transfer-learning",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "KAIST",
          "image": "images/organizations/kaist.png"
        },
        {
          "name": "RLWRLD",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "paper_group_id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "title": "ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning",
      "abstract": "多模态推理需要语言和视觉之间的迭代协调，但目前尚不清楚什么构成有意义的交错思维链。我们认为，文本和图像的思维应该作为互为补充的模态，而不是同构的模态，彼此推动推理。基于这一原则，我们构建了ThinkMorph，这是一个统一模型，在24,000条高质量的交错推理轨迹上进行微调，涵盖了不同视觉参与度的任务。ThinkMorph学习生成渐进的文本-图像推理步骤，这些步骤具体操作视觉内容，同时保持一致的语言逻辑。它在以视觉为中心的基准测试中取得了显著提升（相较于基础模型平均提升34.7%），并对外部领域任务具有良好的泛化能力，能够与更大规模和专有的视觉语言模型相匹配或超越。除了性能，ThinkMorph展现出新兴的多模态智能，包括前所未见的视觉操作技能、推理模式的自适应切换，以及通过多样化的多模态测试时间扩展能力。以上发现为描述统一模型在多模态推理中涌现出的能力指明了有前景的方向。",
      "paper_summary": {
        "summary": "ThinkMorph, a unified model from an international team including researchers from the National University of Singapore and Stanford University, integrates language and vision to generate progressive text-image reasoning steps. The model achieves substantial performance gains on vision-centric tasks, outperforming its base model by 34.74% on average and rivaling larger models, while demonstrating emergent properties like unseen visual manipulations and autonomous mode switching.",
        "originalProblem": [
          "Existing multimodal reasoning models often treat visual input as supplementary to language or rely on indirect, brittle tool-augmented approaches.",
          "Current unified multimodal Chain-of-Thought (CoT) models lack a generalizable method for text and image modalities to genuinely and mutually advance problem-solving.",
          "Multimodal problems requiring active interrogation and manipulation of visual elements (e.g., spatial reasoning) are not adequately addressed by models primarily focused on textual descriptions."
        ],
        "solution": [
          "ThinkMorph is developed as a unified model, fine-tuned from Bagel-7B, to generate seamlessly interleaved sequences of text and image tokens within a single reasoning trace.",
          "A high-quality interleaved training dataset of approximately 24,000 reasoning traces is meticulously curated across four vision-centric tasks, ensuring text and visual manipulations are complementary and verifiable.",
          "The model is trained using a dual-objective optimization with Mean Squared Error (MSE) for image tokens and negative log-likelihood for text tokens, fostering both contextual image generation and coherent verbal logic."
        ],
        "keyInsights": [
          "Deep, interleaved training with complementary text-image data enables unified multimodal models to develop adaptive, human-like 'think-and-sketch' problem-solving strategies.",
          "Interleaving modalities is crucial for tasks demanding precise visual grounding and manipulation, as visual 'thoughts' provide concrete, verifiable actions that textual reasoning alone cannot capture.",
          "Such integrated training leads to emergent multimodal intelligence, including the ability to perform novel visual manipulations, autonomously switch reasoning modes, and achieve robust test-time scaling through diversified multimodal thoughts."
        ],
        "results": [
          "ThinkMorph achieves an average performance gain of 34.74% over its base model (Bagel-7B) on vision-centric benchmarks, with specific increases of 85.84% on Spatial Navigation and 38.75% on Jigsaw Assembly.",
          "The model consistently outperforms unimodal (text-only and vision-only) approaches by an average of 5.33% on vision-centric tasks and rivals or exceeds larger models, outperforming GPT-4o on SAT (52.67% vs. 28.00%).",
          "ThinkMorph exhibits emergent properties: generating accurate, unseen visual manipulations, autonomously switching to text-only reasoning in 5.3% of cases with higher accuracy (81.25%), and achieving an 8.0% accuracy gain on BLINK-J through diversified test-time scaling."
        ]
      },
      "image_url": "image/2510.27492v1.png",
      "universal_paper_id": "2510.27492",
      "metrics": {
        "total_votes": 4,
        "visits_count": {
          "all": 129,
          "last_7_days": 129
        },
        "public_total_votes": 17
      },
      "first_publication_date": "2025-10-30T17:51:38.000Z",
      "publication_date": "2025-10-30T17:51:38.000Z",
      "updated_at": "2025-11-03T04:01:17.389Z",
      "topics": [
        "Computer Science",
        "cs.CV"
      ],
      "organization_info": [
        {
          "name": "University of Washington",
          "image": "images/organizations/uw.png"
        },
        {
          "name": "National University of Singapore",
          "image": "images/organizations/national-university-of-singapore.svg+xml"
        },
        {
          "name": "Stanford University",
          "image": "images/organizations/stanford.png"
        },
        {
          "name": "Zhejiang University",
          "image": "images/organizations/zhejiang.png"
        },
        {
          "name": "The Chinese University of Hong Kong",
          "image": "images/organizations/chinesehongkong.png"
        }
      ],
      "author_info": [],
      "github_stars": 9,
      "github_url": "https://github.com/ThinkMorph/ThinkMorph",
      "distance": 1
    },
    {
      "id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "paper_group_id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "title": "The Denario project: Deep knowledge AI agents for scientific discovery",
      "abstract": "我们介绍了Denario，一个旨在作为科学研究助手的人工智能多智能体系统。Denario可以执行许多不同的任务，例如生成创意、查阅文献、制定研究计划、编写和执行代码、制作图表以及撰写和审查科学论文。该系统具有模块化架构，能够处理特定任务，如生成创意或使用Cmbagent作为深度研究后端进行端到端的科学分析。在这项工作中，我们详细描述了Denario及其模块，并通过展示多个在天体物理学、生物学、生物物理学、生物医学信息学、化学、材料科学、数学物理、医学、神经科学和行星科学等不同科学领域生成的AI论文来说明其能力。Denario在结合不同学科的创意方面也表现出色，我们通过展示一篇将量子物理和机器学习方法应用于天体物理数据的论文来说明这一点。我们报告了领域专家对这些论文进行的评估，他们提供了数值评分和类似审稿的反馈。随后，我们强调了当前系统的优点、缺点和局限性。最后，我们讨论了人工智能驱动研究的伦理影响，并反思这种技术与科学哲学的关系。我们将在此https网址公开发布代码。Denario演示也可以直接在此https网址上运行，完整应用将部署在云上。",
      "paper_summary": null,
      "image_url": "image/2510.26887v1.png",
      "universal_paper_id": "2510.26887",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 80,
          "last_7_days": 80
        },
        "public_total_votes": 11
      },
      "first_publication_date": "2025-10-30T18:00:12.000Z",
      "publication_date": "2025-10-30T18:00:12.000Z",
      "updated_at": "2025-11-03T03:03:45.706Z",
      "topics": [
        "agent-based-systems",
        "agentic-frameworks",
        "cloud-computing",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.MA",
        "human-ai-interaction",
        "information-extraction",
        "ml-systems",
        "reasoning",
        "text-generation",
        "tool-use"
      ],
      "organization_info": [
        {
          "name": "Google DeepMind",
          "image": "images/organizations/deepmind.png"
        },
        {
          "name": "University of Cambridge",
          "image": "images/organizations/university-of-cambridge.svg+xml"
        },
        {
          "name": "Harvard University",
          "image": "images/organizations/harvard.png"
        },
        {
          "name": "University of Oxford",
          "image": "images/organizations/oxford.jpg"
        },
        {
          "name": "LMU Munich",
          "image": null
        },
        {
          "name": "the University of Tokyo",
          "image": "images/organizations/the-university-of-tokyo.jpeg"
        },
        {
          "name": "The University of Texas at Austin",
          "image": "images/organizations/the-university-of-texas-at-austin.jpeg"
        },
        {
          "name": "Cornell University",
          "image": "images/organizations/cornell.png"
        },
        {
          "name": "Harvard Medical School",
          "image": null
        },
        {
          "name": "Johns Hopkins University",
          "image": "images/organizations/jhu.png"
        },
        {
          "name": "University of Arizona",
          "image": "images/organizations/university-of-arizona.png"
        },
        {
          "name": "MIT",
          "image": "images/organizations/mit.jpg"
        },
        {
          "name": "Princeton University",
          "image": "images/organizations/princeton.jpg"
        },
        {
          "name": "Tel-Aviv University",
          "image": null
        },
        {
          "name": "ICREA",
          "image": null
        },
        {
          "name": "Universitat de Barcelona",
          "image": null
        },
        {
          "name": "Flatiron Institute",
          "image": "images/organizations/flatiron-institute.jpeg"
        },
        {
          "name": "University of Virginia",
          "image": "images/organizations/virginia.png"
        },
        {
          "name": "The University of Chicago",
          "image": null
        },
        {
          "name": "Universitat Autònoma de Barcelona",
          "image": null
        },
        {
          "name": "Donostia International Physics Center",
          "image": null
        },
        {
          "name": "University of the Basque Country",
          "image": null
        },
        {
          "name": "Computer Vision Center",
          "image": null
        },
        {
          "name": "ICSC - Centro Nazionale di Ricerca in High Performance Computing, Big Data e Quantum Computing",
          "image": null
        },
        {
          "name": "SISSA - International School for Advanced Studies",
          "image": null
        },
        {
          "name": "Kavli Institute for Cosmology",
          "image": null
        },
        {
          "name": "INAF – Osservatorio Astronomico di Trieste",
          "image": null
        },
        {
          "name": "Steward Observatory",
          "image": null
        },
        {
          "name": "IFPU – Institute for Fundamental Physics of the Universe",
          "image": null
        },
        {
          "name": "Institut de Ciències del Cosmos",
          "image": null
        },
        {
          "name": "INFN – National Institute for Nuclear Physics",
          "image": null
        },
        {
          "name": "Big Data Institute",
          "image": null
        },
        {
          "name": "Infosys Ltd",
          "image": null
        },
        {
          "name": "Boston Children\b\beach Hospital",
          "image": null
        },
        {
          "name": "Ragon Institute of Mass General",
          "image": null
        },
        {
          "name": "MCML - Munich Center for Machine Learning",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 76,
      "github_url": "https://github.com/AstroPilot-AI/Denario",
      "distance": 1
    },
    {
      "id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "paper_group_id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "title": "Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model",
      "abstract": "最近的大型语言模型（LLM）研究经历了从编码器-解码器建模到如今主流的仅解码器建模的结构转变。然而，这一迅速的转变并没有进行严格的比较分析，尤其是在“规模扩展的视角”上，引发了人们对编码器-解码器模型潜力可能被忽视的担忧。为填补这一空白，我们重新审视了编码器-解码器LLM（RedLLM），并结合了最近从仅解码器LLM（DecLLM）得来的新方法。我们在不同模型规模下，对RedLLM（使用前缀语言建模预训练）和DecLLM（使用因果语言建模预训练）进行了全面比较，模型规模范围从约150M到约8B。使用RedPajama V1（1.6T个token）进行预训练，并使用FLAN进行指令微调，我们的实验表明RedLLM展现出令人信服的规模扩展特性和意想不到的强大性能。尽管在预训练阶段，DecLLM整体上更具计算效率，但RedLLM显示出可比的规模扩展能力和上下文长度外推能力。在指令微调后，RedLLM在各种下游任务中取得了可比甚至更好的结果，同时享受了显著更好的推断效率。我们希望我们的发现能够激励更多人重新审视RedLLM，发掘其开发强大而高效的LLM的潜力。",
      "paper_summary": {
        "summary": "Google DeepMind researchers systematically compared modernized encoder-decoder Large Language Models (RedLLM) against decoder-only models (DecLLM) up to 8 billion parameters, finding RedLLM achieves comparable or superior performance post-instruction tuning with significantly better training and inference efficiency. Its superior context length extrapolation also stands out.",
        "originalProblem": [
          "The rapid architectural shift towards decoder-only LLMs lacked a comprehensive, scaling-perspective comparison against encoder-decoder models.",
          "A rigorous analysis of how encoder-decoder architectures scale with increasing model size and computational resources was missing.",
          "Potential benefits of encoder-decoder models regarding efficiency and adaptability for instruction tuning may have been overlooked."
        ],
        "solution": [
          "Developed RedLLM, an enhanced encoder-decoder architecture integrating modern architectural recipes (e.g., Rotary Positional Embedding, SwiGLU) common in successful decoder-only LLMs.",
          "Conducted a systematic, large-scale comparative analysis of RedLLM and DecLLM across five model scales (150M to 8B parameters) on pretraining perplexity, zero-shot, and few-shot tasks.",
          "Assessed adaptability through instruction tuning on the FLAN collection and measured empirical training and inference throughput."
        ],
        "keyInsights": [
          "RedLLM demonstrates remarkable adaptability post-instruction tuning, matching and often surpassing DecLLM's performance on downstream tasks, despite weaker initial pretraining zero/few-shot capabilities.",
          "Encoder-decoder architectures offer substantial advantages in both training and inference efficiency, processing significantly more examples per second than decoder-only models.",
          "RedLLM exhibits stronger context length extrapolation capabilities, showing a smoother and more stable perplexity increase for sequence lengths well beyond pretraining limits (e.g., up to 16K tokens)."
        ],
        "results": [
          "RedLLM's zero-shot and few-shot performance after instruction tuning matched or surpassed DecLLM, particularly when normalized by inference FLOPs.",
          "Empirical efficiency measurements showed RedLLM processed significantly more examples per second during both training and inference across all model sizes.",
          "RedLLM maintained a smoother perplexity curve at extended context lengths (e.g., up to 16K tokens) compared to DecLLM, indicating better generalization to longer sequences."
        ]
      },
      "image_url": "image/2510.26622v1.png",
      "universal_paper_id": "2510.26622",
      "metrics": {
        "total_votes": 4,
        "visits_count": {
          "all": 144,
          "last_7_days": 144
        },
        "public_total_votes": 25
      },
      "first_publication_date": "2025-10-30T15:48:28.000Z",
      "publication_date": "2025-10-30T15:48:28.000Z",
      "updated_at": "2025-10-31T06:53:51.593Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "fine-tuning",
        "inference-optimization",
        "representation-learning",
        "sequence-modeling",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "paper_group_id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "title": "Deep sequence models tend to memorize geometrically; it is unclear why",
      "abstract": "在序列建模中，原子事实的参数化记忆主要被抽象为实体之间共现的暴力查找。我们将这种关联视角与记忆存储的几何视角进行对比。我们首先隔离一个干净且可分析的Transformer推理实例，该实例与将记忆严格视为训练期间指定的局部共现的存储方式不相容。相反，模型必须以某种方式合成其自身的原子事实几何，在所有实体之间编码全球关系，包括那些未共现的实体。这反过来简化了一个涉及$\\ell$-重组合的困难推理任务，使其变为一个易于学习的1步几何任务。\n\n从这一现象中，我们提取出神经嵌入几何的基本方面，这些方面难以解释。我们认为，尽管仅在局部关联上进行优化，这种几何的出现不能简单归因于典型的架构或优化压力。反直觉的是，即使这种几何并不比关联的暴力查找更简洁，它依然被优雅地学习到了。\n\n随后，通过分析与Node2Vec的联系，我们展示了这种几何源于一种谱偏差——与现有理论相反，这种偏差确实是在缺乏各种压力的情况下自然产生的。这一分析还指出了实践者在使Transformer记忆更具几何性的明显空间。我们希望参数化记忆的几何视角能鼓励研究人员重新审视在知识获取、容量、发现和遗忘等领域指导他们的默认直觉。",
      "paper_summary": {
        "summary": "Researchers from CMU and Google Research demonstrate that deep sequence models, including Transformers and Mamba, organize their parametric memory geometrically to encode global relationships for multi-hop reasoning, rather than relying solely on associative lookup. The models achieved high accuracy on an adversarially-designed in-weights path-finding task, revealing that this geometric structuring emerges from local supervision and is not primarily driven by memory capacity constraints or succinctness benefits.",
        "originalProblem": [
          "The fundamental mechanism by which deep sequence models store and utilize 'atomic facts' in their parameters (in-weights) was unclear.",
          "The prevailing 'associative view' of parametric memory, which stores local co-occurrences, struggles to explain advanced reasoning capabilities.",
          "A clear, isolatable instance of implicit in-weights reasoning was needed to probe beyond associative memory explanations."
        ],
        "solution": [
          "Developed an \"in-weights path-star graph\" problem where models memorize fixed graph edges and perform multi-hop path-finding on unseen paths.",
          "Evaluated decoder-only Transformers and Mamba models, along with simpler architectures like Node2Vec, on graphs up to 5x10^4 nodes.",
          "Analyzed token embeddings using cosine distances and UMAP projections to visualize and quantify the internal memory structure."
        ],
        "keyInsights": [
          "Deep sequence models inherently tend to organize parametric memory geometrically, representing global relationships through structured embeddings.",
          "This geometric memory paradigm allows complex compositional reasoning tasks to be reduced to simpler, single-step geometric retrieval processes.",
          "The geometry emerges naturally from local supervision, even when not explicitly trained for global tasks, and is not necessarily more succinct or capacity-efficient than associative memory."
        ],
        "results": [
          "Both Transformers and Mamba achieved up to 100% accuracy on the in-weights path-star task, even on large graphs and long, unseen paths (10 hops), contrasting with their failure on equivalent in-context tasks.",
          "Analysis of token embeddings via UMAP and cosine distances revealed distinct geometric arrangements, with leaf nodes and first-hop nodes from the same path exhibiting close proximity.",
          "Spectral bias was identified as a mechanism for geometry formation in Node2Vec models, occurring naturally without typical theoretical assumptions like low-rank constraints or multi-hop supervision."
        ]
      },
      "image_url": "image/2510.26745v1.png",
      "universal_paper_id": "2510.26745",
      "metrics": {
        "total_votes": 7,
        "visits_count": {
          "all": 239,
          "last_7_days": 239
        },
        "public_total_votes": 35
      },
      "first_publication_date": "2025-10-30T17:40:22.000Z",
      "publication_date": "2025-10-30T17:40:22.000Z",
      "updated_at": "2025-10-31T11:57:55.837Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "embedding-methods",
        "mechanistic-interpretability",
        "model-interpretation",
        "reasoning",
        "representation-learning",
        "sequence-modeling",
        "Statistics",
        "stat.ML",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们推出了Tongyi DeepResearch，一个具有主动性的规模化语言模型，专门设计用于长时间、深入的信息探索研究任务。为了激励自主深入研究能力，Tongyi DeepResearch通过一个端到端的训练框架开发，该框架结合了主动性的中期训练和主动性的后期训练，能够在复杂任务中实现可扩展的推理和信息查找。我们设计了一个高度可扩展的数据合成管道，该管道完全自动化，不依赖于昂贵的人力标注，并支持所有训练阶段。通过为每个阶段构建定制化环境，我们的系统实现了稳定和一致的交互。Tongyi DeepResearch具有305亿个参数，每个token仅激活33亿个参数，在多项主动深度研究基准测试中表现出色，包括人类的最后考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES和xbench-DeepSearch-2510。我们开源了模型、框架和完整解决方案，以助力社区发展。",
      "paper_summary": {
        "summary": "The Tongyi DeepResearch Team from Alibaba Group introduced an open-source model and an end-to-end agentic training framework for autonomous deep research agents, integrating agentic mid-training and post-training with a fully automated data synthesis pipeline. This approach achieves state-of-the-art performance on various benchmarks, including 32.9 on Humanity's Last Exam and 70.9 on GAIA, while activating only 3.3 billion parameters.",
        "originalProblem": [
          "Most frontier Deep Research AI systems are proprietary and closed-source, hindering scientific transparency, reproducibility, and collaborative development.",
          "Developing Deep Research agents faces challenges in scalable training, data scarcity for multi-step reasoning, and managing complex environmental interactions efficiently.",
          "Traditional LLM training lacks agentic inductive bias, making it difficult to effectively cultivate deep reasoning and information-seeking behaviors for complex tasks."
        ],
        "solution": [
          "Developed an end-to-end agentic training framework that includes two stages: agentic mid-training (for foundational agentic bias) and agentic post-training (for refinement via RL).",
          "Implemented a fully automated, highly scalable data synthesis pipeline to generate research-level questions, agentic behaviors, and function-calling data, eliminating reliance on human annotation.",
          "Designed stage-specific, customized environments (Prior World, Simulated, Real-world) to provide stable and consistent interactions throughout the training process, balancing fidelity, cost, and scalability."
        ],
        "keyInsights": [
          "Agentic mid-training serves as a crucial bridge, effectively transitioning LLMs from general pre-training to specialized agentic tasks by cultivating inherent agentic biases.",
          "Synthetic data generation can produce 'super-human' level agent trajectories at scale, overcoming data scarcity and enabling efficient training without human annotation.",
          "A stable reinforcement learning framework, coupled with novel environment designs and algorithmic modifications, ensures robust policy learning and prevents issues like policy collapse."
        ],
        "results": [
          "Achieved state-of-the-art performance across numerous benchmarks, including 32.9 on Humanity's Last Exam (Avg@3), 70.9 on GAIA (Avg@3), and 75.0 on xbench-DeepSearch (Avg@3).",
          "The model, based on Qwen3-30B, demonstrates remarkable efficiency by activating only 3.3 billion parameters per token, allowing for practical deployment.",
          "The 'Heavy Mode' inference strategy further boosts performance, achieving 38.3% on Humanity's Last Exam and 58.3% on BrowseComp (Pass@1) by deploying parallel agents and an integrative synthesis model."
        ]
      },
      "image_url": "image/2510.24701v1.png",
      "universal_paper_id": "2510.24701",
      "metrics": {
        "total_votes": 54,
        "visits_count": {
          "all": 1927,
          "last_7_days": 1927
        },
        "public_total_votes": 158
      },
      "first_publication_date": "2025-10-28T17:53:02.000Z",
      "publication_date": "2025-10-28T17:53:02.000Z",
      "updated_at": "2025-10-29T02:06:57.854Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.IR",
        "cs.LG",
        "cs.MA",
        "data-curation",
        "fine-tuning",
        "information-extraction",
        "lightweight-models",
        "reasoning",
        "synthetic-data",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Alibaba Group",
          "image": "images/organizations/alibaba.png"
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/Alibaba-NLP/DeepResearch",
      "distance": 1
    },
    {
      "id": "019a48b7-d9c3-7625-97a0-47bac9e7457d",
      "paper_group_id": "019a48b7-d9c3-7625-97a0-47bac9e7457d",
      "title": "Higher-order Linear Attention",
      "abstract": "缩放的点积注意力的二次成本是将自回归语言模型扩展到长上下文的主要障碍。线性时间注意力和状态空间模型（SSM）提供了可扩展的替代方案，但通常受到一阶或基于核的近似的限制，这可能限制表达能力。我们引入了高阶线性注意力（HLA），这是一种因果流机制，通过紧凑的前缀充分统计实现更高的交互。在二阶情况下，HLA 维护一个恒定大小的状态，并在不显现任何 $n \\times n$ 矩阵的情况下以线性时间计算每个标记的输出。我们给出了闭合形式的流身份、使用两个额外摘要的严格因果掩蔽变体，以及基于关联扫描的分块并行训练方案，该方案能准确重现串行递归的激活。我们还概述了对三阶及更高阶的扩展。总的来说，这些结果将 HLA 定位为一个原则上可扩展的构建块，结合了类注意力的、数据依赖的混合与现代递归架构的效率。项目页面：这个 https URL。",
      "paper_summary": null,
      "image_url": "image/2510.27258v1.png",
      "universal_paper_id": "2510.27258",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 56,
          "last_7_days": 56
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-31T07:54:37.000Z",
      "publication_date": "2025-10-31T07:54:37.000Z",
      "updated_at": "2025-11-03T07:56:39.747Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "efficient-transformers",
        "lightweight-models",
        "sequence-modeling",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 31,
      "github_url": "https://github.com/yifanzhang-pro/HLA",
      "distance": 1
    },
    {
      "id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "paper_group_id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "title": "$π_\\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models",
      "abstract": "视觉-语言-行动（VLA）模型使得机器人能够从多模态输入中理解并执行复杂任务。尽管近期的工作探讨了使用强化学习（RL）来自动化在扩展监督微调（SFT）中的繁琐数据收集过程，但将大规模RL应用于基于流的VLA（例如，$\\pi_0$，$\\pi_{0.5}$）仍然面临挑战，因为迭代去噪中产生的不可处理的动作对数似然性。我们通过 $\\pi_{\\text{RL}}$ 来解决这个挑战，这是一个用于并行仿真训练基于流的VLA的开源框架。$\\pi_{\\text{RL}}$ 实现了两种RL算法：（1）{Flow-Noise} 将去噪过程建模为具有可学习噪声网络的离散时间马尔可夫决策过程（MDP），以便精确计算对数似然性。（2）{Flow-SDE} 将去噪与智能体-环境交互相结合，形成一个两层的MDP，采用常微分方程（ODE）到随机微分方程（SDE）的转换以实现高效的RL探索。我们在LIBERO和ManiSkill基准上评估了$\\pi_{\\text{RL}}$。在LIBERO上，$\\pi_{\\text{RL}}$ 提升了少样本SFT模型$\\pi_0$和$\\pi_{0.5}$的性能，分别从57.6%提升到97.6%和从77.1%提升到98.3%。在ManiSkill中，我们在320个并行环境中训练$\\pi_{\\text{RL}}$，使$\\pi_0$从41.6%提升到85.7%，而$\\pi_{0.5}$从40.0%提升到84.8%，涵盖了4352个取放任务，展示了在异构仿真下可扩展的多任务RL。总体而言，$\\pi_{\\text{RL}}$ 实现了显著的性能提升和对SFT模型的更强泛化能力，验证了在线RL在基于流的VLA中的有效性。",
      "paper_summary": {
        "summary": "πRL introduces an open-source framework that enables online reinforcement learning (RL) for flow-based Vision-Language-Action (VLA) models, a class previously incompatible with policy gradient methods due to technical challenges. The framework achieves substantial performance gains over supervised fine-tuning baselines, improving success rates on benchmarks like LIBERO and ManiSkill, particularly in few-shot and multi-task scenarios.",
        "originalProblem": [
          "Supervised Fine-Tuning (SFT) for VLA models relies on labor-intensive, costly human demonstration datasets, leading to overfitting and limited generalization.",
          "Existing online Reinforcement Learning (RL) methods are largely incompatible with flow-based VLA architectures.",
          "Flow-based models face challenges with intractable action log-likelihood computation and deterministic action generation, hindering their use with policy gradient RL algorithms like PPO."
        ],
        "solution": [
          "πRL introduces two methods, Flow-Noise and Flow-SDE, to inject stochasticity and enable tractable action log-likelihood computation in flow-based VLAs.",
          "Flow-Noise incorporates a learnable noise network into the denoising process and models it as a one-layer MDP for exact log-likelihood computation with PPO.",
          "Flow-SDE converts the deterministic ODE denoising process into a Stochastic Differential Equation (SDE) and formulates a two-layer MDP with a hybrid ODE-SDE sampling strategy to approximate log-likelihoods."
        ],
        "keyInsights": [
          "The proposed methods effectively bridge the gap between flow-based generative models and online policy gradient RL, making a powerful class of VLAs amenable to environmental interaction.",
          "Flow-Noise, with its learnable noise network and one-layer MDP, slightly outperforms Flow-SDE due to finer control over stochasticity and more efficient data utilization.",
          "Online RL fine-tuning yields substantial benefits for VLA models in few-shot learning and multi-task generalization, often surpassing full-dataset SFT performance by learning from active environmental interaction."
        ],
        "results": [
          "On the LIBERO benchmark, πRL improved π0's average success rate from 57.6% (few-shot SFT) to 97.6% (Flow-Noise) and π0.5's from 77.1% to 98.3%.",
          "For the challenging LIBERO-Long task, πRL boosted π0.5's performance from 43.9% (few-shot SFT) to 94.0%, exceeding the 92.4% achieved by the all-trajectories SFT model.",
          "Demonstrated scalability on ManiSkill MultiTask, increasing π0's success rate from 41.6% to 85.7% and π0.5's from 40.1% to 84.8% on in-distribution tasks involving thousands of combinations."
        ]
      },
      "image_url": "image/2510.25889v1.png",
      "universal_paper_id": "2510.25889",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 159,
          "last_7_days": 159
        },
        "public_total_votes": 22
      },
      "first_publication_date": "2025-10-29T18:37:39.000Z",
      "publication_date": "2025-10-29T18:37:39.000Z",
      "updated_at": "2025-10-31T03:13:44.464Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "deep-reinforcement-learning",
        "distributed-learning",
        "few-shot-learning",
        "fine-tuning",
        "generative-models",
        "multi-task-learning",
        "reinforcement-learning",
        "robotic-control",
        "vision-language-models"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 890,
      "github_url": "https://github.com/RLinf/RLinf",
      "distance": 1
    },
    {
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于大型语言模型的网络代理在信息获取方面展现出巨大的潜力，但它们在长期任务中的有效性受到上下文管理的基本权衡的限制。现有的基于ReAct的代理由于积累了嘈杂的原始历史而遭遇上下文饱和，而每一步固定总结完整历史的方法则可能导致关键信息的不可逆丧失。为了解决这些问题，我们引入了AgentFold，一个以主动上下文管理为中心的新型代理范式，灵感来源于人类认知过程中的回顾性整合。AgentFold将其上下文视为一个动态的认知工作空间，应该积极塑造，而不是一个被动的日志。在每一步中，它学习执行一种“折叠”操作，以在多个层面上管理其历史轨迹：它可以进行细致的压缩，以保留重要的细粒度细节，或进行深入的整合，以抽象出整个多步骤的子任务。它在重要基准测试中的表现令人瞩目：通过简单的监督微调（无需持续预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上达到了36.2%的成绩，在BrowseComp-ZH上达到了47.3%。值得注意的是，这一表现不仅超过或匹配了规模显著更大的开源模型，如DeepSeek-V3.1-671B-A37B，还超越了诸如OpenAI的o4-mini等领先的专有代理。",
      "paper_summary": {
        "summary": "AgentFold presents a novel web agent architecture that employs proactive, multi-scale context management to enable effective reasoning over long-horizon web tasks by dynamically folding interaction histories. This approach achieved state-of-the-art performance for open-source agents on benchmarks like BrowseComp (36.2%) and WideSearch (62.1%), while maintaining a context size 92% smaller than ReAct agents after 100 turns.",
        "originalProblem": [
          "Traditional ReAct agents suffer from context saturation on long-horizon tasks due to their verbose, append-only history, hindering effective reasoning.",
          "Fixed-summarization methods, while concise, risk irreversible loss of crucial fine-grained details during context summarization.",
          "A fundamental trade-off exists between maintaining comprehensive detail and achieving context conciseness, limiting LLM-based web agents' ability to handle complex, prolonged tasks."
        ],
        "solution": [
          "AgentFold introduces a dynamic cognitive workspace partitioning context into user question, tools, multi-scale state summaries, and the latest interaction.",
          "It implements a \"fold\" operation, guided by a learned folding directive, to either granularly condense the latest interaction or deeply consolidate multiple past steps into abstract summaries.",
          "A \"Fold-Generator\" data collection pipeline, utilizing rejection sampling and supervised fine-tuning, trains LLMs (Qwen3-30B-A3B) to perform this explicit context curation."
        ],
        "keyInsights": [
          "Human-inspired retrospective consolidation and active memory sculpting can overcome the context management trade-off in LLM agents.",
          "Explicit, learned context folding directives allow agents to dynamically adapt context granularity, preserving critical details while abstracting irrelevant information.",
          "Training a smaller LLM to perform strategic context management can lead to performance competitive with or superior to much larger models that rely solely on raw context window size."
        ],
        "results": [
          "AgentFold-30B-A3B achieved state-of-the-art performance for open-source agents, scoring 36.2% on BrowseComp (outperforming a 20x larger model) and 62.1% on WideSearch.",
          "Context token count grew sub-linearly, remaining exceptionally concise, with an average context 92% smaller than ReAct agents after 100 turns, yielding significant memory savings.",
          "Demonstrated enhanced long-horizon capabilities, with accuracy continuing to improve up to 256 turns, unlike baselines that saturated and failed due to context overflow."
        ]
      },
      "image_url": "image/2510.24699v1.png",
      "universal_paper_id": "2510.24699",
      "metrics": {
        "total_votes": 10,
        "visits_count": {
          "all": 497,
          "last_7_days": 497
        },
        "public_total_votes": 60
      },
      "first_publication_date": "2025-10-28T17:51:50.000Z",
      "publication_date": "2025-10-28T17:51:50.000Z",
      "updated_at": "2025-10-29T04:59:02.281Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "fine-tuning",
        "reasoning",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/Alibaba-NLP/DeepResearch",
      "distance": 1
    },
    {
      "id": "019a4d54-272a-7d61-9712-1d9963161888",
      "paper_group_id": "019a4d54-272a-7d61-9712-1d9963161888",
      "title": "Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process",
      "abstract": "视觉语言行动（VLA）模型旨在理解自然语言指令和视觉观察，并作为具身代理执行相应的行动。最近的研究将未来图像纳入理解-行动循环中，产生了统一的VLA，能够共同理解、生成和行动——阅读文本和图像，生成未来图像和行动。然而，这些模型要么依赖外部专家进行模态统一，要么将图像生成和行动预测视为独立的过程，从而限制了这些任务之间直接协同的好处。我们的核心理念是通过同步去噪过程共同优化生成和行动，其中迭代精化使得行动在不断且充分的视觉指导下，从初始化演变。我们在提出的统一扩散VLA和联合离散去噪扩散过程（JD3P）中将这一理念付诸实践，这是一个将多模态整合为单一去噪轨迹的联合扩散过程，作为实现理解、生成和行动内在协同的关键机制。我们的模型和理论建立在所有模态的统一标记空间和混合注意力机制之上。我们进一步提出了一个两阶段的训练流程和若干推理时间技术，以优化性能和效率。我们的方法在CALVIN、LIBERO和SimplerEnv等基准上达到最先进的性能，推理速度比自回归方法快4倍，并通过深入的分析和现实世界评估展示了其有效性。我们的项目页面可在此网址找到。",
      "paper_summary": {
        "summary": "Meituan's LongCat-Flash-Omni is a 560-billion-parameter open-source omni-modal model that processes text, image, video, and audio to enable real-time audio-visual interaction. It achieves state-of-the-art performance on various multimodal benchmarks and shows highly competitive results against leading proprietary models.",
        "originalProblem": [
          "Existing unified Vision-Language-Action (VLA) models often rely on separate components for modality unification or employ distinct decoding processes for image generation and action prediction, leading to misalignment and weak coupling.",
          "Many VLA approaches do not fully exploit generated future visual states as explicit guidance for action planning, often treating them as auxiliary tasks or having insufficient synergy during decoding.",
          "Current VLA models, particularly those using autoregressive decoding, suffer from high computational costs and slow inference speeds, limiting their applicability in real-time robotic control."
        ],
        "solution": [
          "UD-VLA introduces a Joint Discrete Denoising Diffusion Process (JD3P) that unifies future image generation and action prediction into a single, synchronous denoising trajectory.",
          "It employs a hybrid attention mechanism within a Transformer, enabling bidirectional attention for tokens within modalities (future images, actions) and causal attention across modalities (actions conditioned on images, but not vice-versa).",
          "A two-stage training pipeline first pre-trains the model for future image generation using a VLM backbone, then fine-tunes it on robot action datasets to jointly optimize image and action generation via JD3P."
        ],
        "keyInsights": [
          "Synchronous and iterative refinement of both future images and actions through JD3P is crucial for achieving deep intrinsic synergy, allowing actions to be precisely guided by anticipated visual consequences.",
          "A hybrid attention mechanism is optimal for VLA models, balancing comprehensive intra-modal interactions with a clear causal flow across modalities to prevent shortcut learning and ensure proper information conditioning.",
          "Explicit future image generation, when deeply integrated and leveraged during inference, provides a powerful 'chain-of-thought' for action planning, converting abstract control problems into more concrete inverse kinematics tasks."
        ],
        "results": [
          "UD-VLA achieved state-of-the-art performance, with an average success length of 4.64 on CALVIN, a 92.7% average success rate on LIBERO, and a 62.5% average success rate on SimplerEnv benchmarks.",
          "The model demonstrated a 4x faster inference speed compared to autoregressive methods, processing 219.3 tokens/s versus 50.2 tokens/s on CALVIN, which is critical for real-time robotic applications.",
          "Ablation studies confirmed the superiority of the hybrid attention mechanism and JD3P, showing that future image generation is more effective than current image reconstruction or no visual generation for action planning."
        ]
      },
      "image_url": "image/2511.01718v1.png",
      "universal_paper_id": "2511.01718",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 17,
          "last_7_days": 17
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-11-03T16:26:54.000Z",
      "publication_date": "2025-11-03T16:26:54.000Z",
      "updated_at": "2025-11-04T05:25:52.042Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.RO",
        "generative-models",
        "image-generation",
        "imitation-learning",
        "multi-modal-learning",
        "robotic-control",
        "robotics-perception",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "Monash University",
          "image": "images/organizations/monash-university.png"
        },
        {
          "name": "Zhejiang University",
          "image": "images/organizations/zhejiang.png"
        },
        {
          "name": "Westlake University",
          "image": "images/organizations/westlake-university.jpeg"
        },
        {
          "name": "HKUST(GZ)",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 3,
      "github_url": "https://github.com/OpenHelix-Team/UD-VLA",
      "distance": 1
    },
    {
      "id": "019a4a47-a458-7a84-9c58-9518130326e8",
      "paper_group_id": "019a4a47-a458-7a84-9c58-9518130326e8",
      "title": "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs",
      "abstract": "评估大型语言模型（LLMs）在需要长期记忆和长上下文推理的任务（例如对话场景中的任务）能力的过程受到现有基准的限制，这些基准往往缺乏叙事连贯性、覆盖范围狭窄，仅测试简单的回忆导向任务。本文提出了一种全面解决这些挑战的方法。首先，我们提出了一种新颖的框架，用于自动生成长达 1000 万个标记、连贯且主题多样的对话，同时附带针对广泛记忆能力的探测问题。基于此，我们构建了 BEAM，一个新基准，包括 100 个对话和 2000 个经过验证的问题。其次，为了提升模型性能，我们提出了 LIGHT——一个受到人类认知启发的框架，为 LLMs 提供三个互补的记忆系统：长期情节记忆、短期工作记忆和一个用于积累突出事实的临时记事本。我们在 BEAM 上的实验表明，即使是具有 100 万标记上下文窗口的 LLM（无论是否增强检索）在对话变长时也面临困难。相反，LIGHT 在各种模型中始终提升性能，攻击基线的平均提升幅度为 3.5%-12.69%，具体取决于基础 LLM。消融研究进一步确认了每个记忆组件的贡献。",
      "paper_summary": {
        "summary": "Researchers from the University of Alberta and UMass Amherst introduce BEAM, a novel benchmark for evaluating long-term memory in large language models through extremely long, coherent conversations. They also propose LIGHT, a cognitive-inspired framework that consistently enhances LLM performance, yielding average improvements of 3.5% to 12.69% and dramatic gains of over 100% on 10M token dialogues compared to baselines.",
        "originalProblem": [
          "Existing benchmarks for LLM long-term memory lack narrative coherence, cover narrow domains, and primarily test simple recall, making it hard to assess real-world complex reasoning.",
          "Even LLMs with large context windows struggle to maintain performance and effectively reason as dialogue length increases significantly.",
          "Current approaches often focus solely on expanding context length, which is insufficient for truly intelligent information management across extended interactions."
        ],
        "solution": [
          "**BEAM Benchmark**: A scalable, automatic framework to generate long (up to 10 million tokens), coherent, and topically diverse conversations, accompanied by probing questions for ten distinct memory abilities.",
          "**LIGHT Framework**: An LLM-agnostic, cognitively-inspired system integrating episodic memory (long-term retrieval), short-term working memory, and a compressed scratchpad for salient facts.",
          "LIGHT's hybrid approach intelligently manages and utilizes information from extensive conversational histories to enhance LLM reasoning capabilities."
        ],
        "keyInsights": [
          "LLMs, even with large context windows, struggle significantly with long-term memory and reasoning in extended, coherent dialogues, indicating that raw token capacity is not a complete solution.",
          "A cognitively-inspired, hybrid memory system like LIGHT can consistently and substantially improve LLM performance on long-context tasks, especially for complex reasoning and extreme dialogue lengths.",
          "The high-quality, comprehensive BEAM benchmark provides a more rigorous tool for evaluating LLM long-term memory, revealing specific strengths and weaknesses across different memory abilities."
        ],
        "results": [
          "State-of-the-art LLMs experienced substantial performance degradation on BEAM as conversation length increased, with performance plummeting in 10 million token scenarios.",
          "The LIGHT framework consistently improved LLM performance, showing average gains of 3.5% to 12.69% over baselines, and dramatic improvements of over 100% for 10M token dialogues.",
          "LIGHT demonstrated the largest relative gains in Summarization (+160.6%), Multi-hop Reasoning (+27.2%), and Preference Following (+76.5%), while Contradiction Resolution remained a challenging task for all models."
        ]
      },
      "image_url": "image/2510.27246v1.png",
      "universal_paper_id": "2510.27246",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 28,
          "last_7_days": 28
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-31T07:29:52.000Z",
      "publication_date": "2025-10-31T07:29:52.000Z",
      "updated_at": "2025-11-03T15:13:20.472Z",
      "topics": [
        "agents",
        "Computer Science",
        "conversational-ai",
        "cs.AI",
        "cs.CL",
        "cs.IR",
        "human-ai-interaction",
        "representation-learning",
        "sequence-modeling",
        "synthetic-data",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/mohammadtavakoli78/BEAM",
      "distance": 1
    },
    {
      "id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "paper_group_id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "title": "Running VLAs at Real-time Speed",
      "abstract": "在本文中，我们展示了如何使用单个消费级GPU以30Hz帧率和最多480Hz轨迹频率运行pi0级多视角VLA。这使得动态和实时任务成为可能，而这些任务之前被认为是大型VLA模型无法实现的。为了实现这一目标，我们引入了一系列策略来消除模型推理中的开销。现实世界的实验表明，使用我们策略的pi0策略在抓取掉落的钢笔任务中取得了100%的成功率。基于这些结果，我们进一步提出了一个用于VLA实时机器人控制的全流式推理框架。代码可在此https URL获得。",
      "paper_summary": {
        "summary": "Researchers from Dexmal and StepFun developed comprehensive optimization strategies enabling state-of-the-art Vision-Language-Action (VLA) models to operate in real-time on a single consumer-grade GPU. Their approach reduced inference latency for a two-view \nlive\n \nlive\n model from 106.5 ms to 27.3 ms, achieving a 100% success rate in a falling pen grasping task with sub-200 ms reaction times.",
        "originalProblem": [
          "Large Vision-Language-Action (VLA) models like \nlive\n \nlive\n suffered from high inference latency (hundreds of milliseconds), making them too slow for dynamic, real-time robotic tasks.",
          "Typical VLA inference speeds necessitated dropping frames from 30 FPS camera streams, impeding a robot's ability to react quickly and precisely to rapidly changing environments.",
          "The inherent latency created a gap between the strong generalization capabilities of large VLA models and their practical deployability in time-sensitive robotic applications."
        ],
        "solution": [
          "Implemented a multi-stage optimization strategy, including CUDA graph mechanisms and computational graph simplification (e.g., QKV projection fusion) to drastically reduce CPU and architectural overheads.",
          "Performed in-depth kernel optimization by manually tuning GEMM tile parameters, fusing gated linear layers, and integrating scalar operations and RMS normalization directly into kernels.",
          "Addressed system-level latencies through optimized image resizing, pinned memory for data transfer, and static CPU buffers, minimizing non-GPU related overheads."
        ],
        "keyInsights": [
          "Systematically tackling both high-level software overheads (e.g., Python interpreter, graph complexity) and low-level hardware utilization (e.g., kernel efficiency) is crucial for pushing large VLA models into real-time performance on consumer GPUs.",
          "Significant performance gains can be achieved through a combination of CUDA graphs, computational graph transformations, and fine-grained kernel tuning via tools like Triton.",
          "The optimized inference enables a ",
          "Full Streaming Inference",
          " paradigm for future hierarchical control systems, allowing concurrent execution of VLM and Action Expert (AE) components at different frequencies."
        ],
        "results": [
          "Reduced inference time for a two-view \nlive\n \nlive\n model on an RTX 4090 GPU from 106.5 ms to 27.3 ms, representing a nearly 4x speedup and successfully meeting the sub-33 ms real-time threshold.",
          "Achieved a 100% success rate over 10 consecutive trials in a real-world falling pen grasping task, with the end-to-end reaction time of the robotic system measured at less than 200 ms.",
          "Demonstrated that the optimized two-view \nlive\n \nlive\n inference at 27.3 ms is remarkably close to the theoretical lower bound of 20.6 ms, indicating high efficiency."
        ]
      },
      "image_url": "image/2510.26742v1.png",
      "universal_paper_id": "2510.26742",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 129,
          "last_7_days": 129
        },
        "public_total_votes": 22
      },
      "first_publication_date": "2025-10-30T17:38:14.000Z",
      "publication_date": "2025-10-30T17:38:14.000Z",
      "updated_at": "2025-10-31T03:05:42.944Z",
      "topics": [
        "Computer Science",
        "cs.RO"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 94,
      "github_url": "https://github.com/Dexmal/realtime-vla",
      "distance": 1
    },
    {
      "id": "019a4d92-4bb5-702b-9c3f-1adb3d571e87",
      "paper_group_id": "019a4d92-4bb5-702b-9c3f-1adb3d571e87",
      "title": "LongCat-Flash-Omni Technical Report",
      "abstract": "我们引入了LongCat-Flash-Omni，这是一款先进的开源全模态模型，具有5600亿个参数，擅长实时音视频交互。通过采用一种受课程启发的逐步训练策略，从简单到越来越复杂的模态序列建模任务，LongCat-Flash-Omni实现了全面的多模态能力，同时保持了强大的单模态能力。LongCat-Flash基于高性能的快捷连接混合专家（MoE）架构，并配备零计算专家，LongCat-Flash-Omni集成了高效的多模态感知和语音重建模块。尽管规模巨大，达到5600亿参数（其中27亿已激活），LongCat-Flash-Omni仍实现了低延迟的实时音视频交互。为了训练基础设施，我们开发了一种模态解耦的并行方案，专门设计用于管理大型多模态训练中固有的数据和模型异质性。这种创新的方法通过保持超过90%的文本训练所实现的吞吐量，展现了卓越的效率。广泛的评估显示，LongCat-Flash-Omni在开源模型中达到了全模态基准的最先进性能。此外，它在包括文本、图像和视频理解，以及音频理解和生成等广泛的模态特定任务上也交付了高度竞争的结果。我们提供了模型架构设计、训练程序和数据策略的全面概述，并将模型开源，以促进社区未来的研究和发展。",
      "paper_summary": {
        "summary": "Meituan's LongCat-Flash-Omni is a 560-billion-parameter open-source omni-modal model that processes text, image, video, and audio to enable real-time audio-visual interaction. It achieves state-of-the-art performance on various multimodal benchmarks and shows highly competitive results against leading proprietary models.",
        "originalProblem": [
          "Integrating diverse modalities (text, audio, image, video) into a single model while preventing performance degradation due to cross-modal heterogeneity.",
          "Developing a unified model capable of both robust offline understanding and low-latency, real-time streaming interaction with audio and video.",
          "Managing the extreme computational complexity and data heterogeneity inherent in training and inferring large-scale (hundreds of billions of parameters) multimodal models efficiently."
        ],
        "solution": [
          "A unified, end-to-end omni-modal architecture integrating LongCat-ViT for vision, a streaming FSMN-based audio encoder/decoder, and a 560B-parameter Shortcut-connected Mixture-of-Experts (ScMoE) LLM backbone.",
          "A progressive, curriculum-inspired six-stage training pipeline over 2.5 trillion tokens, gradually introducing modalities from text-only to full audio-visual integration, followed by SFT and DPO for human alignment and interactivity.",
          "An optimized training infrastructure employing Modality-Decoupled Parallelism (MDP) and an asynchronous streaming inference pipeline with techniques like sparse-dense sampling for high throughput and low-latency real-time interaction."
        ],
        "keyInsights": [
          "Large-scale multimodal training can synergistically enhance core text capabilities, with the model maintaining or improving its foundational text performance after extensive multimodal integration.",
          "The combination of Shortcut-connected Mixture-of-Experts (ScMoE) architecture and Modality-Decoupled Parallelism (MDP) effectively addresses the scale and heterogeneity challenges of training 560B-parameter omni-modal models, achieving high throughput.",
          "A progressive, curriculum-based training strategy is crucial for successfully integrating diverse modalities into a single model, ensuring comprehensive capabilities across all modalities without catastrophic forgetting."
        ],
        "results": [
          "Achieves state-of-the-art performance among open-source omni-modal models on benchmarks like OmniBench, WorldSense, DailyOmni, and UNO-Bench, often rivaling proprietary models such as Gemini-2.5-Pro.",
          "Demonstrates superior performance across various unimodal tasks, including image-to-text, short and long video understanding, speech recognition (lower WER/CER than Gemini-2.5-Pro, GPT-4o-Audio), and audio understanding.",
          "Provides robust real-time audio-visual interaction, achieving the third-highest quantitative score for naturalness and fluency in end-to-end scenarios, excelling in paralinguistic understanding, relevance, and memory."
        ]
      },
      "image_url": null,
      "universal_paper_id": "2511.00279",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 16,
          "last_7_days": 16
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-31T21:58:15.000Z",
      "publication_date": "2025-10-31T21:58:15.000Z",
      "updated_at": "2025-11-04T06:33:44.629Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.DC",
        "cs.LG",
        "cs.MM",
        "cs.SD"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4d7d-e502-7d9e-8226-ada9b3fb2b11",
      "paper_group_id": "019a4d7d-e502-7d9e-8226-ada9b3fb2b11",
      "title": "End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection",
      "abstract": "实现类人灵巧操作仍然是通用机器人面临的一大挑战。尽管视觉-语言-动作（VLA）模型在从演示中学习技能方面展现出潜力，但其可扩展性受限于稀缺的高质量训练数据。现有的数据收集方法存在固有的局限性：手动遥操作对人类操作者造成过载，而自动规划往往产生不自然的动作。我们提出了一种共享自主框架，将控制分为宏观和微观动作。人类操作者通过直观的虚拟现实遥操作引导机器人的手臂姿态，而自主的DexGrasp-VLA策略则利用实时触觉和视觉反馈进行精细的手部控制。这种分工显著减少了认知负担，并有效收集高质量的协调手臂-手部演示。利用这些数据，我们训练了一个增强了我们新型手臂-手部特征增强模块的端到端VLA策略，该模块捕捉宏观和微观运动的独特和共享表征，以实现更自然的协调。我们的纠正遥操作系统通过人机互动循环中的故障恢复实现了持续政策改进。实验表明，我们的框架以最小的人力生成高质量数据，并在包括未见实例在内的多种物体上实现了90%的成功率。全面评估验证了该系统在开发灵巧操作能力方面的有效性。",
      "paper_summary": {
        "summary": "ByteDance Seed researchers introduce a framework for dexterous arm-hand control, combining human VR teleoperation with an autonomous hand VLA policy to efficiently collect data for complex tasks. The system achieves an 88.7% success rate in pick-and-place tasks with diverse objects by employing an Arm-Hand Feature Enhancement module and allows for continuous policy refinement through human-in-the-loop corrections.",
        "originalProblem": [
          "Scarcity of high-quality, large-scale demonstration data for high-Degree-of-Freedom dexterous hands in VLA models.",
          "High cognitive load on human operators during manual teleoperation, leading to inefficient and unscalable data collection.",
          "Automated methods often generate unnatural or suboptimal robot motions, failing to capture nuanced human expert behaviors."
        ],
        "solution": [
          "A Shared Autonomy framework that couples human VR teleoperation for arm control with an autonomous, force-adaptive VLA policy (DexGrasp-VLA) for the hand.",
          "An Arm-Hand Feature Enhancement module within the end-to-end VLA policy that explicitly models distinct arm and hand roles for improved coordination.",
          "A Corrective Human-in-the-Loop Teleoperation system for iterative policy refinement, using failure demonstrations to enhance robustness and adaptation."
        ],
        "keyInsights": [
          "Shared autonomy significantly boosts data collection efficiency (110 trajectories/hour, 25% increase) by reducing human cognitive load while capturing natural human-robot coordinated motions.",
          "Explicitly modeling limb-specific dynamics with the Arm-Hand Feature Enhancement module yields more robust and resilient VLA policies, especially under sensory challenges like visual occlusion.",
          "Tactile sensing, specifically dual-feature integration, is crucial for achieving high success rates (90%) in robust, contact-rich dexterous grasping, enabling compliance and preventing slippage."
        ],
        "results": [
          "Achieved an 88.7% average success rate on pick-and-place tasks across 50 diverse objects, including 85.6% on unseen objects.",
          "The Shared Autonomy framework increased data collection efficiency to 110 trajectories per hour, a 25% improvement over full manual teleoperation.",
          "The Arm-Hand Feature Enhancement module boosted success rates from 88% to 95% on the primary platform and demonstrated greater robustness under visual occlusion (58% vs. 19%)."
        ]
      },
      "image_url": null,
      "universal_paper_id": "2511.00139",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 16,
          "last_7_days": 16
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-31T16:12:02.000Z",
      "publication_date": "2025-10-31T16:12:02.000Z",
      "updated_at": "2025-11-04T06:11:27.618Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.RO"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 4028,
      "github_url": "https://github.com/eliahuhorwitz/Academic-project-page-template",
      "distance": 1
    },
    {
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我改善系统需要与环境互动以实现持续适应。我们介绍了SPICE（自我对弈语料库环境），这是一个强化学习框架，其中单个模型扮演两个角色：挑战者从大语料库中提取文档以生成多样化的推理任务，和推理者解决这些任务。通过对抗动态，挑战者在推理者能力的前沿创建了一个自动课程，而语料库的基础提供了丰富而几乎取之不尽的外部信号，这是持续改进所必需的。与现有的无基础自我对弈方法相比，后者提供的好处更为有限，SPICE在多个模型家族的数学（+8.9%）和一般推理（+9.8%）基准测试中实现了持续的提升。我们的分析揭示了文档基础在SPICE中是一个关键成分，它能够不断生成自身日益具有挑战性的目标并实现这些目标，从而实现持续的自我改进。",
      "paper_summary": {
        "summary": "A research team from FAIR at Meta and NUS developed SPICE, a reinforcement learning framework that enables large language models to continuously enhance their reasoning abilities through adversarial self-play grounded in a vast external document corpus. The method consistently improved performance on mathematical and general reasoning benchmarks by up to 11.9% compared to base models.",
        "originalProblem": [
          "Existing reinforcement learning methods for large language models (LLMs) often rely on human supervision, curated datasets, or domain-specific reward engineering, limiting scalability and generalizability.",
          "Ungrounded self-play techniques (e.g., R-Zero) suffer from hallucination amplification, where factual errors compound, degrading performance over time.",
          "These methods also face information symmetry, making it difficult to create genuinely challenging and diverse tasks when both the problem generator and solver share the same internalized knowledge."
        ],
        "solution": [
          "SPICE employs a single LLM that dynamically switches between two adversarial roles: a Challenger that generates tasks and a Reasoner that solves them, both grounded in an external document corpus.",
          "The Challenger samples passages from a vast corpus to create diverse, multi-format tasks (multiple-choice or free-form) with verifiable gold answers, ensuring factual accuracy.",
          "A variance-based curriculum reward guides the Challenger to generate tasks at an optimal difficulty level, specifically targeting a 50% Reasoner success rate, to drive continuous learning."
        ],
        "keyInsights": [
          "Corpus grounding is critical for sustained self-improvement in LLMs, effectively addressing hallucination amplification and providing an inexhaustible, verifiable knowledge source.",
          "The adversarial co-evolution of the Challenger and Reasoner components creates an automatic and adaptive curriculum that continuously pushes the Reasoner's capabilities.",
          "Establishing information asymmetry, where the Reasoner solves tasks without access to the source document, ensures genuine challenge and fosters authentic reasoning development.",
          "The variance-based reward mechanism for the Challenger is crucial for calibrating task difficulty, optimizing the learning rate for the Reasoner across various tasks."
        ],
        "results": [
          "SPICE consistently outperformed state-of-the-art self-play methods and strong supervised baselines, achieving average performance gains from +5.7% to +11.9% across multiple base models (Qwen3, OctoThinker families).",
          "The framework demonstrated broad applicability, improving mathematical reasoning benchmarks by an average of +8.9% and general reasoning tasks by +9.8%, indicating generalizable skill development.",
          "Ablation studies confirmed the critical role of corpus grounding and adversarial co-training, showing that their absence leads to significantly lower performance (e.g., 40.7% vs. 43.9% on Qwen3-4B-Base without grounding).",
          "Qualitative analysis revealed that Challengers evolved to generate complex, multi-step tasks from the same documents, while Reasoners developed highly structured and sophisticated problem-solving patterns."
        ]
      },
      "image_url": "image/2510.24684v1.png",
      "universal_paper_id": "2510.24684",
      "metrics": {
        "total_votes": 12,
        "visits_count": {
          "all": 424,
          "last_7_days": 424
        },
        "public_total_votes": 59
      },
      "first_publication_date": "2025-10-28T17:46:16.000Z",
      "publication_date": "2025-10-28T17:46:16.000Z",
      "updated_at": "2025-10-29T02:24:46.100Z",
      "topics": [
        "Computer Science",
        "continual-learning",
        "cs.CL",
        "generative-models",
        "information-extraction",
        "multi-agent-learning",
        "reasoning",
        "reinforcement-learning",
        "self-supervised-learning",
        "text-generation"
      ],
      "organization_info": [
        {
          "name": "National University of Singapore",
          "image": "images/organizations/national-university-of-singapore.svg+xml"
        },
        {
          "name": "Meta",
          "image": "images/organizations/meta.png"
        },
        {
          "name": "FAIR at Meta",
          "image": null
        }
      ],
      "author_info": [
        {
          "id": "01975168-d67a-7df4-9b30-a27d352bdb39",
          "username": "chuanyang-jin",
          "realName": "Chuanyang Jin",
          "avatar": null,
          "institution": "New York University",
          "googleScholarId": "OZeqpLIAAAAJ",
          "reputation": 15,
          "weeklyReputation": 0,
          "verified": false,
          "role": "user",
          "orcidId": null,
          "githubUsername": null,
          "xUsername": null,
          "linkedinUsername": null,
          "blueskyUsername": null,
          "publicEmail": null
        }
      ],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a390a-6398-7cfc-8161-edd680691708",
      "paper_group_id": "019a390a-6398-7cfc-8161-edd680691708",
      "title": "The End of Manual Decoding: Towards Truly End-to-End Language Models",
      "abstract": "“端到端”标签对于大型语言模型（LLMs）是一个误称。实际上，它们依赖于一个不可微分的解码过程，这需要对温度和top-p等超参数进行繁琐的手动调整。本文介绍了AutoDeco，这是一种新颖的架构，通过学习控制自己的解码策略，实现真正的“端到端”生成。我们在标准变换器上增加了轻量级头部，在每一个步骤中动态预测上下文特定的温度和top-p值，并与下一个token的概率值一同输出。这一方法将解码转变为一个参数化的token级过程，使模型能够在一次前向传播中自我调节其采样策略。\n\n通过在八个基准上的广泛实验，我们展示了AutoDeco不仅显著优于默认的解码策略，而且性能达到了基于“破解测试集”方法的oracle调优基线的水平——这是任何静态方法的实际上限。重要的是，我们发现了一种基于指令的解码控制的突出现能力：模型学会理解自然语言命令（例如，“以低随机性生成”），并在每个token的基础上调整其预测的温度和top-p，为可引导和交互式的LLM解码开辟了新的范式。",
      "paper_summary": {
        "summary": "AutoDeco introduces an augmented language model architecture that dynamically predicts decoding parameters such as temperature and top-p at each generation step, achieving truly end-to-end text generation. This framework demonstrates superior performance across diverse benchmarks with minimal computational overhead and exhibits an emergent capability for natural language-based control over output style.",
        "originalProblem": [
          "Existing \"end-to-end\" Large Language Models (LLMs) rely on non-differentiable, manual decoding processes with static, pre-defined hyperparameters.",
          "Optimal decoding parameter settings are highly task-dependent and require laborious, costly manual tuning, hindering efficient development.",
          "Static decoding methods cannot adapt to dynamic, token-level requirements within a single generation, leading to inherently suboptimal outputs."
        ],
        "solution": [
          "AutoDeco augments standard transformer architectures with lightweight prediction heads that dynamically determine optimal temperature and top-p values from the model's hidden state at each decoding step.",
          "A novel, differentiable \"soft\" top-p mechanism is introduced during training to enable end-to-end optimization by overcoming the non-differentiable nature of standard top-p sampling.",
          "The AutoDeco heads are trained efficiently on top of pre-trained LLMs, freezing the base model parameters, using standard cross-entropy loss and de-biasing operations for robustness."
        ],
        "keyInsights": [
          "Integrating decoding parameter prediction directly into the model's differentiable forward pass allows the LLM to self-regulate its generation, realizing a truly end-to-end pipeline.",
          "The \"soft\" top-p mechanism provides a trainable pathway for adapting sampling strategies, eliminating the need for external, heuristic control.",
          "AutoDeco learns a fundamental \"meta-skill\" for adaptive text generation, exhibiting strong zero-shot generalization across diverse tasks despite being trained on specific domains."
        ],
        "results": [
          "AutoDeco consistently outperforms static decoding strategies (Greedy Search, Default Sampling) across eight diverse benchmarks, achieving performance comparable to or exceeding an oracle-tuned baseline.",
          "The proposed architecture adds negligible computational overhead, with only a 1-2% increase in inference latency and minimal memory footprint, making it highly practical.",
          "A groundbreaking emergent capability allows AutoDeco to interpret high-level natural language commands (e.g., \"more innovative\") and spontaneously adjust its token-level decoding parameters accordingly, a behavior solidified through targeted training."
        ]
      },
      "image_url": "image/2510.26697v1.png",
      "universal_paper_id": "2510.26697",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 183,
          "last_7_days": 183
        },
        "public_total_votes": 29
      },
      "first_publication_date": "2025-10-30T17:01:43.000Z",
      "publication_date": "2025-10-30T17:01:43.000Z",
      "updated_at": "2025-10-31T06:52:53.528Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "fine-tuning",
        "generative-models",
        "human-ai-interaction",
        "instruction-tuning",
        "meta-learning",
        "test-time-inference",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 3,
      "github_url": "https://github.com/Zacks917/AutoDeco",
      "distance": 1
    },
    {
      "id": "019a4921-1f51-7775-b01b-ead888f87ea2",
      "paper_group_id": "019a4921-1f51-7775-b01b-ead888f87ea2",
      "title": "Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals",
      "abstract": "分布匹配蒸馏（DMD）将基于评分的生成模型蒸馏为高效的一步生成器，而无需与其教师的采样轨迹进行一一对应。然而，有限的模型容量导致一步蒸馏模型在复杂生成任务中表现不佳，例如在文本到视频生成中合成复杂的物体运动。直接将DMD扩展到多步蒸馏会增加内存使用和计算深度，导致不稳定和效率降低。尽管前人工作提出了随机梯度截断作为潜在解决方案，我们观察到这显著降低了多步蒸馏模型的生成多样性，使其降至一步模型的水平。为了解决这些限制，我们提出了分阶段DMD，这是一种将阶段性蒸馏与专家混合（MoE）理念结合的多步蒸馏框架，降低学习难度，同时增强模型容量。分阶段DMD建立在两个关键理念之上：渐进式分布匹配和子区间内的评分匹配。首先，我们的模型将信噪比（SNR）范围划分为子区间，逐步将模型精炼到更高的SNR水平，以更好地捕捉复杂分布。接下来，为了确保每个子区间内的训练目标准确，我们进行了严格的数学推导。我们通过蒸馏最先进的图像和视频生成模型验证了分阶段DMD，包括Qwen-Image（200亿参数）和Wan2.2（280亿参数）。实验结果表明，分阶段DMD在保持关键生成能力的同时，比DMD更好地保留了输出多样性。我们将发布我们的代码和模型。",
      "paper_summary": {
        "summary": "Phased DMD introduces a few-step distillation framework for accelerating large diffusion models, leveraging a phase-wise approach with a novel subinterval score matching objective. The method effectively preserves generative diversity and core capabilities of multi-billion parameter models like Qwen-Image and Wan2.2, reducing inference steps without compromising quality.",
        "originalProblem": [
          "One-step diffusion model distillation methods offer speed but lack the capacity for complex generative tasks, often yielding low-quality outputs.",
          "Directly extending Distribution Matching Distillation (DMD) to multiple steps leads to training instability, high memory usage, and deep computational graphs.",
          "The Stochastic Gradient Truncation Strategy (SGTS), while improving stability, severely compromises the generative diversity of multi-step distilled models."
        ],
        "solution": [
          "A progressive distribution matching framework partitions the diffusion process into 'k' phases, training a specialized expert generator for each phase, naturally forming a Mixture-of-Experts (MoE) architecture.",
          "A novel, theoretically derived score matching objective is used within subintervals, addressing the challenge of unavailable clean data samples (x_0) during intermediate distillation steps.",
          "The method employs reverse nested SNR intervals for sampling, which empirically improves generation quality and robustness during phase-wise training."
        ],
        "keyInsights": [
          "Decomposing the diffusion process into distinct phases, each handled by a specialized 'expert,' enhances model capacity and stability, aligning with the temporal dynamics of diffusion models across SNR levels.",
          "A mathematically rigorous score matching objective can be derived for subintervals, allowing for effective distribution matching even when the original clean data is not directly accessible.",
          "Utilizing reverse nested intervals for noise injection during training is critical for maintaining high generation quality and structural integrity, outperforming disjoint interval approaches."
        ],
        "results": [
          "Phased DMD achieved superior generative diversity, with DINOv3 Cosine Similarity scores of 0.782 for Wan2.1-T2V-14B (vs. 0.826 for SGTS) and LPIPS distances of 0.544 (vs. 0.521 for SGTS).",
          "The distilled models effectively retained motion dynamics and camera control capabilities in video generation, with Phased DMD scoring 7.57 in Optical Flow (T2V) compared to 3.23 for SGTS.",
          "Phased DMD successfully preserved prompt adherence and high-quality text rendering capabilities from the Qwen-Image model."
        ]
      },
      "image_url": "image/2510.27684v1.png",
      "universal_paper_id": "2510.27684",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 19,
          "last_7_days": 19
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-31T17:55:10.000Z",
      "publication_date": "2025-10-31T17:55:10.000Z",
      "updated_at": "2025-11-03T09:51:38.834Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "ensemble-methods",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "knowledge-distillation",
        "lightweight-models",
        "model-compression",
        "optimization-methods",
        "video-understanding"
      ],
      "organization_info": [
        {
          "name": "Beihang University",
          "image": "images/organizations/beihang-university.png"
        },
        {
          "name": "SenseTime Research",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4d9b-eed2-7b77-a036-53431925e9c6",
      "paper_group_id": "019a4d9b-eed2-7b77-a036-53431925e9c6",
      "title": "Towards Robust Mathematical Reasoning",
      "abstract": "找到合适的北极星指标对于推动基础模型的数学推理能力至关重要，尤其是考虑到现有的评估要么过于简单，要么仅专注于获得正确的短答案。为了解决这些问题，我们提出了IMO-Bench，这是一个经过顶尖专家小组审核的高级推理基准套件，专门针对国际数学奥林匹克（IMO）的水平，这是年轻数学家的最负盛名的赛事。IMO-AnswerBench首先在400个多样化的奥林匹克问题上对模型进行测试，这些问题具有可验证的短答案。IMO-Proof Bench是下一阶段对证明写作能力的评估，涵盖了基本和高级的IMO级别问题，并提供详细的评分指南以促进自动评分。这些基准在我们在2025年IMO上与Gemini Deep Think（Luong和Lockhart，2025）取得金牌级别的历史成就中发挥了关键作用。我们的模型在IMO-AnswerBench上取得了80.0%的得分，在高级IMO-Proof Bench上获得了65.7%的得分，分别比最佳的非Gemini模型高出6.9%和42.4%。我们还展示了使用Gemini推理构建的自动评分器与人类评估之间的良好相关性，并构建了IMO-GradingBench，其包含1000个关于证明的人类评分，以推动长答案的自动评估的进一步发展。我们希望IMO-Bench能帮助社区推动稳健的数学推理，并在这个URL上发布。",
      "paper_summary": {
        "summary": "Google DeepMind developed IMO-Bench, a benchmark suite designed to assess advanced mathematical reasoning in large language models through problem-solving, proof writing, and proof grading tasks. The Gemini Deep Think (IMO Gold) model achieved 80.0% accuracy on robustified problems and 65.7% on challenging proof-writing tasks.",
        "originalProblem": [
          "Existing mathematical reasoning benchmarks (e.g., GSM8K, MATH, AIME) are approaching saturation, limiting their utility in differentiating advanced model capabilities.",
          "Many current benchmarks primarily rely on final answer matching, which can lead to models guessing or memorizing without demonstrating robust multi-step reasoning.",
          "A lack of comprehensive evaluation frameworks for assessing deeper mathematical understanding, such as the ability to generate and rigorously evaluate proofs."
        ],
        "solution": [
          "Introduces IMO-Bench, a comprehensive suite comprising three benchmarks: IMO-AnswerBench for robust problem-solving, IMO-Proof Bench for rigorous proof writing, and IMO-GradingBench for proof evaluation.",
          "IMO-AnswerBench utilizes 400 Olympiad problems across four categories, with explicit 'robustification' techniques (paraphrasing, numerical changes) to prevent data memorization.",
          "IMO-Proof Bench features 60 IMO-level proof problems, divided into basic and advanced sets, which are primarily evaluated by human experts on a 0-7 point scale, mirroring traditional IMO grading."
        ],
        "keyInsights": [
          "Robust \"north-star metrics\" like IMO-Bench are crucial for driving the advancement of foundation models' mathematical reasoning beyond the limitations of saturated, simpler benchmarks.",
          "Evaluating proof-writing capabilities, instead of just final answers, provides a more accurate and comprehensive assessment of a model's underlying reasoning process and logical argumentation.",
          "Problem robustification techniques effectively prevent models from achieving high scores through data memorization, ensuring that genuine reasoning is tested."
        ],
        "results": [
          "Gemini Deep Think (IMO Gold) achieved a state-of-the-art 80.0% accuracy on IMO-AnswerBench and 65.7% on the advanced set of IMO-Proof Bench, significantly outperforming other frontier models.",
          "Robustification on IMO-AnswerBench consistently led to a drop in model performance (e.g., Gemini 2.5 Pro dropped 3.5%), validating its effectiveness in preventing memorization.",
          "Automated verifiers, AnswerAutoGrader and ProofAutoGrader (both built on Gemini 2.5 Pro), demonstrated high correlation with human expert evaluations, with AnswerAutoGrader achieving 98.9% accuracy."
        ]
      },
      "image_url": "image/2511.01846v1.png",
      "universal_paper_id": "2511.01846",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 14,
          "last_7_days": 14
        },
        "public_total_votes": 2
      },
      "first_publication_date": "2025-11-03T18:53:02.000Z",
      "publication_date": "2025-11-03T18:53:02.000Z",
      "updated_at": "2025-11-04T06:44:16.210Z",
      "topics": [
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "data-curation",
        "reasoning",
        "reasoning-verification",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/google-deepmind/superhuman",
      "distance": 1
    },
    {
      "id": "019a4782-1d96-7c98-8bc1-843eaa2fdb30",
      "paper_group_id": "019a4782-1d96-7c98-8bc1-843eaa2fdb30",
      "title": "Chain-of-Thought Hijacking",
      "abstract": "大型推理模型（LRMs）通过在推理时分配更多计算资源来实现更高的任务表现，先前的研究表明这种规模化推理也可能通过改善拒绝能力来增强安全性。然而，我们发现情况正好相反：相同的推理可以被用来绕过安全防护。我们引入了思维链劫持，这是对推理模型的一种越狱攻击。该攻击通过用长序列的无害解谜推理填充有害请求。在HarmBench中，思维链劫持在Gemini 2.5 Pro、GPT o4 mini、Grok 3 mini和Claude 4 Sonnet上的攻击成功率分别达到了99%、94%、100%和94%——远远超过了以往针对LRMs的越狱方法。为了理解我们攻击的有效性，我们进行了机制分析，结果显示中间层编码了安全检查的强度，而后期层编码了验证结果。长的无害思维链通过将注意力转移 away from harmful tokens 稀释了这两个信号。通过这项分析识别的注意力头的定向消融因果性地降低了拒绝，确认了它们在安全子网络中的作用。这些结果表明，最易于解释的推理形式——显式思维链——在与最终答案提示结合时本身也可以成为越狱的媒介。我们发布了提示、输出和评判决策，以促进复制研究。",
      "paper_summary": {
        "summary": "Researchers uncovered \"Chain-of-Thought Hijacking,\" a new vulnerability in Large Reasoning Models where long, benign reasoning sequences weaken internal safety mechanisms. The attack achieves near-perfect success rates on leading proprietary models by diluting refusal signals and shifting attention away from malicious payloads.",
        "originalProblem": [
          "A prevalent assumption in LRM research suggests that long reasoning sequences, like Chain-of-Thought (CoT), inherently enhance model safety and robustness against harmful prompts.",
          "Existing jailbreak attacks primarily focus on prompt rewriting or encoding, overlooking CoT as a potential attack surface that could systematically weaken safety mechanisms.",
          "There was a lack of mechanistic understanding of how advanced reasoning processes like CoT might interact with and potentially undermine LLM safety protocols."
        ],
        "solution": [
          "Introduced \"Chain-of-Thought Hijacking\" (CoT Hijacking), a black-box, prompt-based jailbreak attack that embeds malicious instructions within extensive, benign reasoning prefaces.",
          "Developed \"Seduction,\" an automated pipeline for generating and iteratively refining these CoT Hijacking prompts to optimize attack success.",
          "Employed mechanistic interpretability techniques, including refusal direction analysis, activation interventions, refusal component quantification, and attention pattern analysis, to diagnose the attack's internal mechanisms."
        ],
        "keyInsights": [
          "Contrary to expectations, longer Chain-of-Thought sequences can weaken, rather than strengthen, Large Reasoning Model safety mechanisms, creating a systematic vulnerability.",
          "Refusal behavior in LRMs is governed by a fragile, low-dimensional \"refusal direction\" in the activation space, which can be diluted.",
          "CoT Hijacking operates by \"refusal dilution,\" where extended benign reasoning shifts attention away from harmful payloads and weakens the safety signal in the model's deeper layers."
        ],
        "results": [
          "CoT Hijacking achieved exceptional Attack Success Rates (ASR) of 94-100% on frontier proprietary models, including Gemini 2.5 Pro (99%), ChatGPT o4 mini (94%), Grok 3 mini (100%), and Claude 4 Sonnet (94%).",
          "Experiments demonstrated a direct correlation between the length of benign CoT sequences and increased ASR, with longer reasoning leading to higher attack success.",
          "Mechanistic analysis showed that removing the identified \"refusal direction\" increased ASR to 91%, while adding it drastically reduced ASR to 1%, confirming its causal role in safety."
        ]
      },
      "image_url": null,
      "universal_paper_id": "2510.26418",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 55,
          "last_7_days": 55
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-30T12:10:03.000Z",
      "publication_date": "2025-10-30T12:10:03.000Z",
      "updated_at": "2025-11-03T02:18:20.950Z",
      "topics": [
        "adversarial-attacks",
        "adversarial-robustness",
        "attention-mechanisms",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cybersecurity",
        "explainable-ai",
        "mechanistic-interpretability",
        "reasoning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/gentlyzhao/Hijacking",
      "distance": 1
    },
    {
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLM）的推理能力方面展现了显著的潜力。然而，RL 在 LLM 上的成功很大程度上依赖于人工策划的数据集和可验证的奖励，这限制了其可扩展性和普适性。近期的自我对弈 RL 方法受到游戏和围棋范式成功的启发，旨在无需人工标注数据就提高 LLM 的推理能力。然而，他们的方法主要依赖于有反馈的具体环境（例如，Python 解释器或游戏引擎）；将其扩展到一般领域仍然具有挑战性。为了解决这些挑战，我们提出了多智能体演变（MAE），这是一个能够使 LLM 在解决数学、推理和一般知识问答等多样任务中自我演化的框架。MAE 的核心设计基于一个由单个 LLM 实例化的三元相互作用的智能体（提议者、求解者、评估者），并应用强化学习来优化他们的行为。提议者生成问题，求解者尝试解决方案，评估者在共同演化的过程中对两者进行评估。在 Qwen2.5-3B-Instruct 上的实验表明，MAE 在多个基准上实现了平均提升 4.54%。这些结果突显了 MAE 作为一种可扩展、高效利用数据的方法，能够在最小依赖人工策划监督的情况下增强 LLM 的一般推理能力。",
      "paper_summary": {
        "summary": "A framework called Multi-Agent Evolve (MAE) enables large language models (LLMs) to improve their reasoning abilities through a co-evolutionary process involving self-generated questions, answers, and evaluations, eliminating the need for human-curated data or external verifiable rewards. The method significantly outperforms existing baselines, with the MAE (half reference) setting achieving an overall average score of 59.87 on diverse benchmarks using a Qwen2.5-3B-Instruct model.",
        "originalProblem": [
          "Current reinforcement learning methods for LLMs rely heavily on expensive human-curated datasets and domain-specific verifiable rewards, limiting their scalability and generality.",
          "Self-play paradigms for LLMs often require 'grounded environments' that provide objective feedback, which is difficult to apply in open-ended, general reasoning domains.",
          "Integrating LLM-as-a-Judge mechanisms into robust, general self-improvement loops for continuous training has faced challenges in stability and implementation complexity."
        ],
        "solution": [
          "MAE instantiates three interacting roles (Proposer, Solver, Judge) from a single, shared LLM backbone, forming a closed self-improving loop for generating training data and reward signals.",
          "The Proposer generates questions, the Solver answers them, and the Judge evaluates both questions and answers, providing numerical reward signals without human ground truth.",
          "All three roles are trained simultaneously using Task-Relative REINFORCE++, performing synchronized parameter updates on the uniform shared model to foster co-evolution and stable learning."
        ],
        "keyInsights": [
          "A 'beyond-zero-sum' reward design, incentivizing the Proposer to generate challenging yet solvable questions for the Solver while ensuring intrinsic question quality, is crucial for stable and effective co-evolution.",
          "Maintaining high-quality training data through a Judge-based question quality filtering mechanism is essential to prevent degradation of the question dataset and ensure sustained learning.",
          "A balanced approach that combines exploring novel self-generated questions with referencing an existing, diverse distribution of unlabeled seed questions (MAE (half reference)) is most effective for general reasoning improvement."
        ],
        "results": [
          "MAE (zero), starting with only 16 self-generated questions, improved the base model's overall average score from 55.33 to 58.51, surpassing the strong AZR baseline (57.72) without relying on verifiable environments.",
          "All MAE variants, which do not use ground-truth answers, consistently outperformed Supervised Fine-Tuning (SFT) on the same seed dataset, demonstrating the framework's data efficiency and robustness.",
          "The MAE (half reference) setting achieved the highest overall average score of 59.87, showing strong performance on both in-distribution (68.95) and out-of-distribution (43.96) benchmarks."
        ]
      },
      "image_url": "image/2510.23595v3.png",
      "universal_paper_id": "2510.23595",
      "metrics": {
        "total_votes": 21,
        "visits_count": {
          "all": 890,
          "last_7_days": 884
        },
        "public_total_votes": 92
      },
      "first_publication_date": "2025-10-27T17:58:02.000Z",
      "publication_date": "2025-10-30T04:45:55.000Z",
      "updated_at": "2025-10-28T18:42:08.153Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "fine-tuning",
        "multi-agent-learning",
        "reasoning",
        "reinforcement-learning",
        "self-supervised-learning",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [
        {
          "id": "0199c19a-427b-7683-a107-dc0d38a7f05b",
          "username": "siqi-zhu",
          "realName": "Siqi Zhu",
          "avatar": null,
          "institution": "UIUC",
          "googleScholarId": "eYBAyoIAAAAJ",
          "reputation": 15,
          "weeklyReputation": 0,
          "verified": false,
          "role": "user",
          "orcidId": "0009-0008-7709-0311",
          "githubUsername": "",
          "xUsername": "realagi25",
          "linkedinUsername": "",
          "blueskyUsername": "",
          "publicEmail": "siqizhu4@illinois.edu"
        }
      ],
      "github_stars": 0,
      "github_url": "https://github.com/ulab-uiuc/Multi-agent-Evolve",
      "distance": 1
    },
    {
      "id": "019a4cb1-35ce-7b94-b9ef-59267ca6f51c",
      "paper_group_id": "019a4cb1-35ce-7b94-b9ef-59267ca6f51c",
      "title": "Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation",
      "abstract": "图形布局生成是一个日益增长的研究领域，重点在于生成审美愉悦的布局，从海报设计到文档等多种形式。虽然近期的研究探索了如何结合用户约束来指导布局生成，但这些约束通常需要复杂的规范，从而降低了可用性。我们提出了一种创新的方法，利用用户提供的草图作为直观约束，并通过实证研究展示了这种新指导方法的有效性，将草图到布局的问题确立为一个有前景的研究方向，目前这一方向尚未得到充分探索。为了解决草图到布局的问题，我们提出了一种基于多模态变压器的解决方案，将草图和内容资产作为输入来生成高质量的布局。由于从人工注释者那里收集草图训练数据以训练我们的模型是非常昂贵的，我们引入了一种新颖且高效的方法，以大规模合成生成训练草图。我们在三个公开可用的数据集上训练和评估我们的模型：PubLayNet、DocLayNet和SlidesVQA，结果表明其性能优于最先进的基于约束的方法，同时提供了更直观的设计体验。为了促进未来草图到布局的研究，我们为上述公共数据集发布了约20万个合成生成的草图。数据集可在此URL获取。",
      "paper_summary": {
        "summary": "Researchers at Google DeepMind and EPFL developed a system that generates graphic layouts guided by user sketches and multimodal content, achieving over 40% higher Maximum IoU compared to prior methods. Their approach leverages a fine-tuned Vision-Language Model, trained using a novel method for synthetically generating large-scale sketch-layout datasets.",
        "originalProblem": [
          "Existing layout generation methods often rely on complex and unintuitive user constraints like precise numerical inputs or verbose textual descriptions.",
          "There is a scarcity of large-scale, paired sketch-layout datasets needed to train Vision-Language Models for sketch-guided layout generation.",
          "Applying sketch-based guidance to general, multimodal graphic layouts using advanced VLMs was under-explored due to data limitations."
        ],
        "solution": [
          "The problem is formulated as a code generation task, where layouts are encoded as structured protocol buffer strings, ensuring interpretable and editable outputs.",
          "A multimodal Vision-Language Model (PaLIGemma 3B) is fine-tuned to accept both visual inputs (hand-drawn sketches, image assets) and textual inputs (prompts, text content) for layout generation.",
          "A scalable, two-step synthetic sketch generation pipeline is introduced to compose vast datasets of sketch-layout pairs from existing ground-truth layouts and a small collection of human-drawn primitives."
        ],
        "keyInsights": [
          "Sketches offer a superior time-performance trade-off for guiding layout generation compared to other constraint methods, demonstrating their intuitiveness and efficiency.",
          "Synthetically generated sketches are effective proxies for human-drawn sketches in VLM training, validated by comparable model performance on both synthetic and human inputs.",
          "Content-awareness, achieved by incorporating actual image and text assets, is crucial for improving layout quality and semantic coherence, outperforming content-agnostic models."
        ],
        "results": [
          "The method achieved over 40% improvement in Maximum IoU compared to state-of-the-art constraint-based baselines (e.g., LayoutPrompter) across PubLayNet, DocLayNet, and SlidesVQA datasets.",
          "Models trained on synthetic sketches showed a minimal distribution shift, with comparable mIoU on synthetic (0.592) and human-produced (0.590) sketches on DocLayNet.",
          "A new Content Ordering Score (COS) was introduced, with the content-aware model achieving higher scores (e.g., 0.69 on PubLayNet), indicating improved preservation of narrative flow."
        ]
      },
      "image_url": "image/2510.27632v1.png",
      "universal_paper_id": "2510.27632",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 12,
          "last_7_days": 12
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-10-31T17:05:10.000Z",
      "publication_date": "2025-10-31T17:05:10.000Z",
      "updated_at": "2025-11-04T02:27:53.422Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "generative-models",
        "human-ai-interaction",
        "image-generation",
        "multi-modal-learning",
        "representation-learning",
        "synthetic-data",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 4,
      "github_url": "https://github.com/google-deepmind/sketch_to_layout",
      "distance": 1
    }
  ],
  "page": 0
};