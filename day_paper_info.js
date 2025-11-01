const papersData = {
  "papers": [
    {
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们介绍了Tongyi DeepResearch，这是一种主动的大型语言模型，专门设计用于长期、深度的信息获取研究任务。为了激励自主深度研究能力，Tongyi DeepResearch通过一个端到端的训练框架进行开发，结合了主动中期训练和主动后期训练，使其能够在复杂任务中实现可扩展的推理和信息获取。我们设计了一个高度可扩展的数据合成管道，完全自动化，无需依赖高成本的人类注释，并支持所有训练阶段。通过为每个阶段构建定制化环境，我们的系统实现了稳定和一致的互动。Tongyi DeepResearch拥有305亿总参数，每个令牌仅激活33亿参数，在一系列主动深度研究基准测试中实现了最先进的性能，包括人类的最后考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES和xbench-DeepSearch-2510。我们开源了模型、框架和完整解决方案，以支持社区发展。",
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
        "total_votes": 37,
        "visits_count": {
          "all": 1359,
          "last_7_days": 1359
        },
        "public_total_votes": 104
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
      "id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "paper_group_id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "title": "Defeating the Training-Inference Mismatch via FP16",
      "abstract": "大语言模型（LLMs）的强化学习（RL）微调常常因训练和推理策略之间的数值不匹配而出现不稳定。虽然之前的工作试图通过算法修正或工程对齐来缓解这个问题，但我们表明其根本原因在于浮点数精度本身。广泛采用的BF16尽管具有较大的动态范围，但引入了大量的四舍五入误差，打破了训练与推理之间的连续性。在本研究中，我们证明，仅仅回退到FP16便有效消除了这种不匹配。这个变化简单，现代框架完全支持，只需少量代码更改，并且不需要修改模型架构或学习算法。我们的结果表明，使用FP16可以统一地实现更稳定的优化、更快的收敛和更强的性能，适用于各种任务、算法和框架。我们希望这些发现能激励人们更广泛地重新考虑核算精度的权衡，以进行RL微调。",
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
        "total_votes": 11,
        "visits_count": {
          "all": 244,
          "last_7_days": 244
        },
        "public_total_votes": 35
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
      "abstract": "现代的大型语言模型（LLM）主要通过显式文本生成方式进行“思考”，如链式思维（CoT），这将推理推迟到训练后，并未充分利用预训练数据。我们提出并开源了Ouro，命名源自递归的乌鲁波罗斯，这是一系列预训练的循环语言模型（LoopLM），它们通过(i) 潜在空间中的迭代计算，(ii) 对学习深度分配的熵正则化目标，以及(iii) 扩展到77T标记，在预训练阶段构建推理。Ouro 1.4B和2.6B模型在广泛的基准测试中展现了优越的性能，甚至可以与高达12B的最新SOTA LLM结果相匹配。通过控制实验，我们证明这种优势并非源于知识容量的增加，而是源于更出色的知识操控能力。我们还展示了LoopLM产生的推理痕迹与最终输出的对齐程度高于显式CoT。我们希望我们的结果展示了LoopLM作为推理时代一种新颖扩展方向的潜力。我们的模型可以在此链接找到：这个http URL。",
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
        "total_votes": 15,
        "visits_count": {
          "all": 568,
          "last_7_days": 568
        },
        "public_total_votes": 47
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
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们引入了Kimi Linear，这是一种混合线性注意力架构，它首次在各种场景下（包括短上下文、长上下文和强化学习（RL）扩展机制）以公平的比较方式超越了全注意力架构。其核心是Kimi Delta Attention（KDA），这是一个表现力强的线性注意力模块，扩展了Gated DeltaNet，具有更精细的门控机制，能够更有效地利用有限的有限状态RNN内存。我们定制的分块算法通过一种专门的对角加低秩（DPLR）转移矩阵变体实现了高硬件效率，与普通DPLR公式相比，大幅降低了计算量，同时保持了与经典增量规则的一致性。\n\n我们预训练了一个具有30亿激活参数和480亿总参数的Kimi Linear模型，基于KDA和多头潜在注意力（MLA）的分层混合。我们的实验表明，在相同的训练方案下，Kimi Linear在所有评估任务中显著超越了全MLA，同时将KV缓存使用量减少了多达75%，在1M上下文下达到了最多6倍的解码吞吐量。这些结果表明，Kimi Linear可以作为全注意力架构的替代品，具有更出色的性能和效率，包括处理更长输入和输出长度的任务。\n\n为了支持进一步的研究，我们开源了KDA内核和vLLM实现，并发布了预训练和指令调优的模型检查点。",
      "paper_summary": {
        "summary": "Kimi Linear introduces a hybrid attention architecture that integrates Kimi Delta Attention (KDA) with periodic full attention layers, matching or surpassing the quality of full attention models across various benchmarks. This architecture significantly enhances inference efficiency, achieving up to 6x faster decoding throughput and a 75% reduction in KV cache usage for million-token contexts.",
        "originalProblem": [
          "Standard self-attention mechanisms in LLMs suffer from quadratic time complexity and linearly growing Key-Value (KV) cache memory, hindering performance for long-context and reinforcement learning applications.",
          "Prior linear attention methods often struggled to match the expressivity and quality of full attention, limiting their utility in complex tasks.",
          "Scaling and evaluating hybrid attention architectures robustly across diverse benchmarks remained a challenge for industrial-grade LLM deployment."
        ],
        "solution": [
          "A new linear attention module, Kimi Delta Attention (KDA), is developed, featuring channel-wise gating for finer memory control and a hardware-efficient Diagonal-Plus-Low-Rank (DPLR) variant with a custom chunkwise algorithm.",
          "Kimi Linear, a hybrid architecture, integrates KDA layers with periodic Multi-Head Latent Attention (MLA) layers in a 3:1 ratio to balance efficiency and global information flow.",
          "All MLA layers are designed with No Positional Encoding (NoPE), allowing KDA layers to dynamically handle positional information and recency bias."
        ],
        "keyInsights": [
          "Kimi Delta Attention's channel-wise gating mechanism enables finer-grained control over recurrent memory, selectively forgetting irrelevant information while preserving crucial details more effectively than previous methods.",
          "Customizing a Diagonal-Plus-Low-Rank (DPLR) matrix for KDA, combined with a bespoke chunkwise-parallel algorithm, yields substantial hardware efficiency gains and operator speedup.",
          "A hybrid architecture uniformly interleaving KDA and full attention layers (3:1 ratio) provides the optimal balance of quality and efficiency across various tasks.",
          "Using No Positional Encoding (NoPE) in full attention layers within the hybrid architecture, and relying on KDA for dynamic positional encoding, improves long-context extrapolation and robustness."
        ],
        "results": [
          "Kimi Linear consistently outperformed full attention and other efficient attention baselines across a wide range of tasks, including language understanding, reasoning, long-context retrieval, and reinforcement learning.",
          "Inference efficiency for 1M context sequences improved dramatically, showing up to 6x faster decoding throughput and a 75% reduction in KV cache memory usage.",
          "The architecture achieved approximately 1.16x computational efficiency compared to full attention under compute-optimal training conditions.",
          "Kimi Linear demonstrated superior long-context capabilities, achieving a RULER score of 94.8 at 1M context length, and exhibited faster convergence and higher accuracy in Math RL training."
        ]
      },
      "image_url": "image/2510.26692v1.png",
      "universal_paper_id": "2510.26692",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 186,
          "last_7_days": 186
        },
        "public_total_votes": 24
      },
      "first_publication_date": "2025-10-30T16:59:43.000Z",
      "publication_date": "2025-10-30T16:59:43.000Z",
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
      "id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "paper_group_id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "title": "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning",
      "abstract": "大型语言模型（LLMs）通常在需要多步推理的问题上表现不佳。对于小规模的开源模型，具有可验证奖励的强化学习（RLVR）在经过多次尝试后，即使正确的解决方案很少被采样也会失败，而监督微调（SFT）则往往由于严格的逐令模仿而导致对长示范的过拟合。为了解决这个问题，我们提出了监督强化学习（SRL），一个将问题解决重新定义为生成一系列逻辑“行动”的框架。SRL训练模型在执行每个行动之前生成内部推理独白。它根据模型的行动与从SFT数据集中提取的专家行动之间的相似性，以逐步方式提供更顺畅的奖励。这种监督即使在所有回合都不正确的情况下也提供了更丰富的学习信号，同时鼓励灵活的推理，受到专家示范的引导。因此，SRL使小模型能够学习以前无法通过SFT或RLVR学习的难题。此外，使用SRL初始化训练，然后再用RLVR进行精细化，可以获得最强的总体性能。除了推理基准测试，SRL还能够有效推广到自主软件工程任务，确立其作为面向推理的LLMs的强大而多功能的训练框架。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 151,
          "last_7_days": 151
        },
        "public_total_votes": 18
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
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们介绍Emu3.5，一种大规模的多模态世界模型，能够本地预测视觉和语言中的下一个状态。Emu3.5通过统一的下一个标记预测目标，端到端地在包含超过10万亿个标记的视觉-语言交错数据语料库上进行预训练，主要数据来源是互联网视频的顺序帧和文字稿。该模型自然接收交错的视觉-语言输入，并生成交错的视觉-语言输出。Emu3.5还通过大规模的强化学习进行后训练，以增强多模态推理和生成能力。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐标记解码转换为双向并行预测，将每幅图像的推理速度提高约20倍，而不牺牲性能。Emu3.5展示了强大的原生多模态能力，包括长期视觉-语言生成、任意到图像（X2I）生成，以及复杂的文本丰富图像生成。它还展现了可泛化的世界建模能力，能够在多种场景和任务中实现时空一致的世界探索和开放世界实体操作。作为对比，Emu3.5在图像生成和编辑任务中表现达到与Gemini 2.5 Flash Image（Nano Banana）相当的水平，并在一系列交错生成任务中展示了更优的结果。我们在这个URL上开源Emu3.5，以支持社区研究。",
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
        "total_votes": 8,
        "visits_count": {
          "all": 151,
          "last_7_days": 151
        },
        "public_total_votes": 23
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
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于LLM的网络代理在信息搜索方面展现出巨大的潜力，但在长时间跨度任务中的有效性受到上下文管理的基本权衡的制约。现有的基于ReAct的代理在积累嘈杂的原始历史时面临上下文饱和的问题，而在每一步固定总结整个历史的方法则有不可逆转地丧失关键细节的风险。为了解决这些问题，我们推出了AgentFold，这是一种以主动上下文管理为中心的新型代理范式，灵感来自于人类认知过程中的回顾整合。AgentFold将其上下文视为一个动态的认知工作空间，积极进行雕塑，而不是被动地填充日志。在每一步中，它学习执行一个“折叠”操作，在多个层面管理其历史轨迹：它可以进行细致的浓缩，以保留重要的、细致的细节，或进行深层整合，以抽象出整个多步骤子任务。在显著的基准测试中的结果令人瞩目：通过简单的监督微调（没有持续的预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上达到了36.2%，在BrowseComp-ZH上达到了47.3%。值得注意的是，这一表现不仅超越或匹配了规模大得惊人的开源模型，如DeepSeek-V3.1-671B-A37B，还超过了领先的专有代理，如OpenAI的o4-mini。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 276,
          "last_7_days": 276
        },
        "public_total_votes": 35
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
      "id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "paper_group_id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "title": "The Era of Agentic Organization: Learning to Organize with Language Models",
      "abstract": "我们设想了一个新的AI时代，称为主动组织，代理通过协作和并行工作解决复杂问题，从而实现超越个体智能的结果。为了实现这一愿景，我们引入了异步思维（AsyncThink），作为一种与大型语言模型进行推理的新范式，它将内部思维过程组织成可以并行执行的结构。具体而言，我们提出了一种思维协议，其中组织者动态地分配子查询给工作者，合并中间知识，并生成连贯的解决方案。更重要的是，这个协议中的思维结构可以通过强化学习进一步优化。实验表明，AsyncThink的推理延迟比并行思维低28%，同时提高了数学推理的准确性。此外，AsyncThink能够将其学习到的异步思维能力进行泛化，有效应对未见过的任务而无需额外训练。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 85,
          "last_7_days": 85
        },
        "public_total_votes": 10
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
      "id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "paper_group_id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "title": "Uniform Discrete Diffusion with Metric Path for Video Generation",
      "abstract": "连续空间视频生成迅速发展，而离散方法由于误差积累和长时间上下文不一致而滞后。在本工作中，我们重新审视离散生成建模，并提出了均匀离散扩散与度量路径（URSA），这是一个简单而强大的框架，填补了可扩展视频生成与连续方法之间的鸿沟。URSA的核心将视频生成任务设定为离散时空标记的迭代全局精炼。它集成了两个关键设计：线性度量路径和依赖分辨率的时间步移机制。这些设计使URSA能够高效扩展到高分辨率图像合成和长时段视频生成，同时所需的推理步骤显著减少。此外，我们引入了一种异步时间微调策略，统一了单一模型内的多种任务，包括插值和图像到视频生成。在具有挑战性的视频和图像生成基准上进行的大量实验表明，URSA始终优于现有离散方法，并达到与最先进的连续扩散方法可比的性能。代码和模型可在此链接获取。",
      "paper_summary": {
        "summary": "URSA presents a uniform discrete diffusion framework that incorporates a metric probability path for video generation, enabling iterative global refinement in discrete token space. This framework achieves performance competitive with state-of-the-art continuous diffusion models across text-to-video, image-to-video, and text-to-image benchmarks, while enhancing scalability and multi-task capabilities.",
        "originalProblem": [
          "Existing discrete generative methods (autoregressive, masked diffusion) suffer from error accumulation and lack long-context consistency, particularly critical for video generation.",
          "These methods typically lack the iterative global refinement capability found in continuous diffusion models, limiting visual quality and spatiotemporal coherence.",
          "Scaling discrete models efficiently to high-resolution images and long-duration videos often requires a prohibitive number of inference steps and struggles with global context modeling."
        ],
        "solution": [
          "URSA introduces a uniform discrete diffusion process with a linearized metric path derived from token embedding distances, establishing a controlled relationship between timestep and perturbation.",
          "It employs resolution-dependent timestep shifting to adapt perturbation levels to varying sequence lengths and complexities, ensuring appropriate scaling for different resolutions.",
          "An asynchronous timestep scheduling strategy assigns independent continuous timesteps to each frame in a video, enabling a single model to learn diverse multi-task generation objectives like text-to-video, image-to-video, and extrapolation."
        ],
        "keyInsights": [
          "Iterative global refinement is critical for discrete generative models to achieve high fidelity and temporal coherence in video synthesis, mirroring its importance in continuous diffusion models.",
          "A carefully designed linearized metric probability path in discrete space, with appropriate scheduling, is essential for effective model convergence and performance, establishing a linear relationship akin to continuous diffusion.",
          "While increasing model size improves semantic performance, current discrete vision tokenizers might bottleneck overall generation quality, suggesting a need for more advanced tokenization strategies."
        ],
        "results": [
          "URSA achieves a total VBench score of 82.4 for text-to-video generation, outperforming discrete baselines and rivaling several Sora-like continuous models, particularly in semantic content and dynamic degree (81.4).",
          "For image-to-video generation, URSA scores 86.2 on VBench++, demonstrating comparable performance to specialized continuous models like SEINE and DynamiCrafter, and showing strong 'Camera Motion' capabilities (37.6).",
          "The model exhibits robust zero-shot generalization, successfully performing video extrapolation from 4 seconds to 40 seconds and generating coherent transitions between specified start and end frames."
        ]
      },
      "image_url": "image/2510.24717v1.png",
      "universal_paper_id": "2510.24717",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 257,
          "last_7_days": 257
        },
        "public_total_votes": 29
      },
      "first_publication_date": "2025-10-28T17:59:57.000Z",
      "publication_date": "2025-10-28T17:59:57.000Z",
      "updated_at": "2025-10-29T02:15:51.179Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "fine-tuning",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "representation-learning",
        "sequence-modeling",
        "unsupervised-learning"
      ],
      "organization_info": [
        {
          "name": "Chinese Academy of Sciences",
          "image": "images/organizations/chinese-academy-of-sciences.jpeg"
        },
        {
          "name": "Zhejiang University",
          "image": "images/organizations/zhejiang.png"
        },
        {
          "name": "Beijing Academy of Artificial Intelligence",
          "image": null
        },
        {
          "name": "National Laboratory of Pattern Recognition, CASIA",
          "image": null
        },
        {
          "name": "Key Laboratory of Intelligent Information Processing, ICT, CAS",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 13,
      "github_url": "https://github.com/baaivision/URSA",
      "distance": 1
    },
    {
      "id": "019a390a-6398-7cfc-8161-edd680691708",
      "paper_group_id": "019a390a-6398-7cfc-8161-edd680691708",
      "title": "The End of Manual Decoding: Towards Truly End-to-End Language Models",
      "abstract": "“端到端”标签对大型语言模型（LLMs）来说是一个误称。实际上，它们依赖于一种不可微的解码过程，需要对温度和top-p等超参数进行繁琐的手动调节。本文介绍了AutoDeco，这是一种新颖的架构，通过学习控制自己的解码策略，实现真正的“端到端”生成。我们在标准变换器上增强了轻量级头部，实时动态预测针对上下文的温度和top-p值，以及下一个标记的logits。这种方法将解码转化为一个参数化的、按标记级别的过程，使得模型能够在一次前向传播中自我调节采样策略。\n\n通过对八个基准的广泛实验，我们证明了AutoDeco不仅显著优于默认解码策略，还达到了与从“破解测试集”得出的经过oracle调优的基线相当的性能，这为任何静态方法设定了一个实际的上限。至关重要的是，我们发现了一种基于指令的解码控制的突现能力：模型学习理解自然语言命令（例如，“以低随机性生成”），并在按标记的基础上调整其预测的温度和top-p，为可引导和交互式的LLM解码开辟了新的范式。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 56,
          "last_7_days": 56
        },
        "public_total_votes": 9
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
      "id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "paper_group_id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "title": "Deep sequence models tend to memorize geometrically; it is unclear why",
      "abstract": "在序列建模中，原子事实的参数化记忆主要被抽象为实体之间共现的强制查找。我们将这种关联视角与记忆存储的几何视角进行对比。我们首先隔离出一个干净且可分析的Transformer推理实例，该实例与记忆严格作为训练期间指定的局部共现存储不兼容。相反，模型必须以某种方式合成其自身的原子事实几何，编码所有实体之间的全球关系，包括那些未共现的实体。这反过来简化了一个涉及 $\\ell$ 次组合的困难推理任务，变成一个易于学习的一步几何任务。\n\n从这一现象中，我们提取出神经嵌入几何中的一些基本方面，这些方面难以解释。我们认为，尽管仅仅是优化局部关联的几何形状的出现，不能简单归因于典型的架构或优化压力。从直觉上看，优雅的几何形状在不比强制查找关联更简洁的情况下仍然被学习到。\n\n然后，通过分析与Node2Vec的联系，我们展示了几何是如何源自谱偏差的——与主流理论相反，这种偏差确实在缺乏各种压力的情况下自然产生。这一分析还为实践者指明了让Transformer记忆更加几何化的可见空间。我们希望这种参数化记忆的几何视角能够鼓励研究人员重新审视指导知识获取、容量、发现和遗忘等领域的默认直觉。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 50,
          "last_7_days": 50
        },
        "public_total_votes": 9
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
      "id": "019a362d-6c5a-7a32-96c6-e60ae4941758",
      "paper_group_id": "019a362d-6c5a-7a32-96c6-e60ae4941758",
      "title": "What Really Matters in Matrix-Whitening Optimizers?",
      "abstract": "最近出现了一系列优化器，它们以不同方式近似相同的“矩阵白化”变换。在这项工作中，我们系统性地拆解了这些优化器，旨在剖析解释性能的关键组成部分。在各项调优超参数下，所有类型的矩阵白化方法都可靠地超越了逐元素对应方法，例如Adam。矩阵白化通常与谱下降相关——然而，实验表明性能提升并不*仅仅是通过准确的谱归一化来解释的*——具体而言，SOAP在每一步上显示出最大的增益，即使Muon在最陡峭的谱下降方向上下降得更准确。相反，我们认为矩阵白化有两个目的，而矩阵白化的方差适应成分则是解释这一性能差距的被忽视的因素。实验表明，方差适应版本的优化器始终优于其符号下降对应方法，包括Muon的自适应版本。我们进一步消融了方差适应策略，发现虽然前瞻式近似效果不如预期，但低秩方差估计器能够有效降低内存成本，而不损失性能。",
      "paper_summary": {
        "summary": "Research from UC Berkeley systematically deconstructs matrix-whitening optimizers, revealing that their performance gains are not solely from accurate spectral normalization but are crucially driven by variance adaptation. The study, conducted on GPT-2 training, demonstrates that these methods can achieve equivalent validation loss in 66% to 83% fewer gradient steps than Adam when thoroughly tuned.",
        "originalProblem": [
          "A systematic understanding of the core components driving the performance gains of matrix-whitening optimizers (e.g., Shampoo, Muon, SOAP) was lacking.",
          "Prior comparisons of advanced optimizers were often confounded by auxiliary implementation details or uneven hyperparameter tuning, making it difficult to pinpoint true performance drivers.",
          "The exact roles of spectral normalization versus other adaptive mechanisms in matrix-based optimizers for efficient deep learning training were unclear."
        ],
        "solution": [
          "A rigorous, controlled experimental setup was established, training a GPT-2 Transformer on OpenWebText with consistent data ordering, random seeds, and initial parameters.",
          "Each optimizer underwent thorough, independent hyperparameter tuning across learning rate, weight decay, momentum, and variance coefficients to ensure fair comparisons.",
          "Optimizers were systematically benchmarked and ablated based on a conceptual framework differentiating spectral normalization and variance adaptation, using specific pairs to isolate effects."
        ],
        "keyInsights": [
          "The performance gains of matrix-whitening optimizers are not exclusively attributable to accurate spectral normalization; methods with less precise spectral normalization can still achieve top performance.",
          "Variance adaptation, similar to the elementwise scaling in Adam, is a critical and often underestimated component that consistently boosts performance when integrated with matrix-based spectral transformations.",
          "Spectral normalization and variance adaptation can be effectively decoupled, with variance adaptation providing benefits even when applied after orthogonalization, indicating distinct roles."
        ],
        "results": [
          "Thoroughly tuned matrix-whitening optimizers consistently outperformed elementwise methods like Adam, achieving equivalent validation loss in 66% to 83% fewer gradient steps.",
          "Variance-adapted variants of optimizers (e.g., SOAP over SPlus, AdaMuon over Muon) consistently showed superior performance compared to their purely spectral or sign-descent counterparts.",
          "Low-rank factorization of elementwise variance estimators effectively reduced memory costs for variance adaptation without significant performance loss, and in some cases, improved it."
        ]
      },
      "image_url": "image/2510.25000v1.png",
      "universal_paper_id": "2510.25000",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 44,
          "last_7_days": 44
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-28T21:59:49.000Z",
      "publication_date": "2025-10-28T21:59:49.000Z",
      "updated_at": "2025-10-30T17:32:17.882Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "optimization-methods",
        "parameter-efficient-training"
      ],
      "organization_info": [
        {
          "name": "UC Berkeley",
          "image": "images/organizations/berkeley.png"
        }
      ],
      "author_info": [],
      "github_stars": 4,
      "github_url": "https://github.com/kvfrans/matrix-whitening",
      "distance": 1
    },
    {
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLMs）的推理能力方面展示了显著的潜力。然而，RL在LLMs中的成功高度依赖于人工策划的数据集和可验证的奖励，这限制了它们的可扩展性和通用性。最近的自我对战RL方法受到游戏和围棋中这一范式成功的启发，旨在增强LLMs的推理能力，而无需人类注释的数据。然而，他们的方法主要依赖于有反馈的具体环境（例如Python解释器或游戏引擎），将其扩展到一般领域仍然面临挑战。为了解决这些挑战，我们提出了多智能体进化（MAE），这是一个框架，允许LLMs在解决包括数学、推理和一般知识问答等多样任务时自我进化。MAE的核心设计基于三个相互作用的智能体（提议者、求解者、评判者），它们由单一的LLM实例化，并应用强化学习来优化它们的行为。提议者生成问题，求解者尝试解决方案，评判者在共同进化的过程中评估两者。对Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试上实现了平均4.54%的改进。这些结果凸显了MAE作为一种可扩展、数据高效的方法，以最小依赖人工策划的监督来增强LLMs的一般推理能力。",
      "paper_summary": {
        "summary": "Multi-Agent Evolve (MAE) enables large language models (LLMs) to improve their reasoning and problem-solving abilities across general domains through a self-supervised, multi-agent co-evolutionary framework. The approach allows a 3B-parameter LLM to increase its overall average accuracy from 55.33% to 58.51% without human annotation and even outperform supervised fine-tuning with ground truth.",
        "originalProblem": [
          "Current LLM training with Reinforcement Learning heavily relies on costly human-curated datasets and verifiable reward signals, limiting scalability and generalizability.",
          "Existing self-play methods for LLMs often require \"grounded environments\" to provide explicit feedback, restricting their application to open-ended language and reasoning tasks.",
          "A lack of mechanisms for continuous, autonomous enhancement of LLM reasoning abilities across diverse tasks without human supervision."
        ],
        "solution": [
          "A framework instantiating three co-evolving LLM agents (Proposer, Solver, Judge) from a single backbone model.",
          "Domain-agnostic self-rewarding mechanisms are used, where the Judge agent evaluates questions and answers without external ground truth.",
          "A continuous training loop with question quality filtering and synchronized updates using Task-Relative REINFORCE++ drives the self-evolutionary process."
        ],
        "keyInsights": [
          "LLMs can autonomously achieve self-improvement in general domains by generating their own curriculum and feedback, eliminating reliance on human supervision or external verifiers.",
          "Generating questions of \"desirable difficulty\" (challenging but feasible) is crucial for optimal and continuous performance enhancement.",
          "The comprehensive interaction and co-evolution of distinct agent roles (Proposer, Solver, Judge) are essential for framework stability and achieving optimal self-improvement."
        ],
        "results": [
          "MAE improved the Qwen2.5-3B-Instruct model's overall average accuracy from 55.33% to 58.51% without any reference questions, outperforming the AZR baseline.",
          "MAE variants, including those without ground truth, significantly surpassed Supervised Fine-Tuning on the same dataset with ground truth, with MAE (half reference) achieving 59.87% overall average.",
          "The framework demonstrated stable performance over 250 training steps, with continuous addition of high-quality questions preventing dataset degradation."
        ]
      },
      "image_url": "image/2510.23595v1.png",
      "universal_paper_id": "2510.23595",
      "metrics": {
        "total_votes": 15,
        "visits_count": {
          "all": 649,
          "last_7_days": 649
        },
        "public_total_votes": 63
      },
      "first_publication_date": "2025-10-27T17:58:02.000Z",
      "publication_date": "2025-10-27T17:58:02.000Z",
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
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/ulab-uiuc/Multi-agent-Evolve",
      "distance": 1
    },
    {
      "id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "paper_group_id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "title": "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents",
      "abstract": "关于大规模监督微调AI代理的公共研究结果仍然相对稀少，因为收集代理训练数据面临独特的挑战。在本研究中，我们认为瓶颈并不是缺乏基础数据源，而是各种数据分散在异构格式、工具和接口中。为此，我们引入了代理数据协议（ADP），这是一种轻量级表示语言，充当不同格式的代理数据集与统一的代理训练管道之间的“中介语”。ADP的设计足够表达，以捕捉各种任务，包括API/工具使用、浏览、编码、软件工程和一般代理工作流程，同时在解析和训练时保持简单，而无需针对每个数据集进行工程化。在实验中，我们将13个现有的代理训练数据集统一转换为ADP格式，并将标准化的ADP数据转化为多个代理框架的训练准备格式。我们对这些数据进行了监督微调（SFT），并展示了相对于相应基本模型约20%的平均性能提升，并在标准编码、浏览、工具使用和研究基准上实现了最先进或接近最先进的性能，无需特定领域的调优。所有代码和数据均已公开发布，希望ADP能够帮助降低标准化、可扩展和可重复的代理训练的门槛。",
      "paper_summary": {
        "summary": "Carnegie Mellon University and collaborating researchers introduced the Agent Data Protocol (ADP), a standardized framework unifying diverse agent training datasets. This protocol enables more effective and scalable fine-tuning of LLM agents, achieving approximately 20% average performance gain over base models and enhancing cross-task generalization.",
        "originalProblem": [
          "Agent training data is highly fragmented across datasets, utilizing inconsistent formats, action spaces, and observation structures.",
          "The heterogeneity of existing data sources hinders the ability to combine and leverage them for large-scale supervised fine-tuning (SFT) of LLM agents.",
          "A significant engineering overhead exists for integrating new datasets or adapting agent frameworks, as custom conversion scripts are required for each dataset-agent pair."
        ],
        "solution": [
          "Developed the Agent Data Protocol (ADP), a lightweight, expressive representation language to standardize diverse agent training datasets.",
          "Implemented ADP using Pydantic schemas, defining core `Trajectory` objects composed of standardized `Actions` (API, Code, Message) and `Observations` (Text, Web).",
          "Created a three-stage conversion pipeline that transforms raw datasets into ADP-standardized format, then converts them into SFT-ready formats tailored for specific agent frameworks."
        ],
        "keyInsights": [
          "A systematic analysis of 13 diverse datasets, unified via ADP, revealed inherent characteristics such as varied trajectory lengths, domain-specific action patterns, and high reasoning coverage (typically >90%).",
          "Unifying diverse datasets through ADP enables the training of more generalizable LLM agents, outperforming models fine-tuned on single-domain data and mitigating negative transfer effects.",
          "The ADP framework demonstrably reduces the engineering effort for integrating datasets and agent frameworks, transforming a quadratic (D x A) integration problem into a linear (D + A) one."
        ],
        "results": [
          "Fine-tuning LLMs on the standardized ADP dataset yielded an average performance gain of approximately 20% over base models across benchmarks like SWE-Bench, WebArena, AgentBench OS, and GAIA.",
          "On SWE-Bench (Verified), ADP-trained Qwen-2.5-7B-Coder-Instruct improved accuracy from 0.4% to 20.2% (SWE-Agent) and 2.8% to 20.4% (OpenHands), with the 32B model achieving 40.3% (SWE-Agent).",
          "ADP reduced the engineering cost for dataset-to-agent integration, illustrated by a reduction from an estimated ~489,200 Lines of Code without ADP to ~12,592 LOC with ADP for 13 datasets and 100 harnesses."
        ]
      },
      "image_url": "image/2510.24702v1.png",
      "universal_paper_id": "2510.24702",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 261,
          "last_7_days": 261
        },
        "public_total_votes": 30
      },
      "first_publication_date": "2025-10-28T17:53:13.000Z",
      "publication_date": "2025-10-28T17:53:13.000Z",
      "updated_at": "2025-10-29T14:44:09.882Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "data-curation",
        "fine-tuning",
        "instruction-tuning",
        "ml-systems",
        "tool-use",
        "training-orchestration",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "paper_group_id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "title": "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender",
      "abstract": "在推荐系统中，扩展特征交互模块（例如，Wukong、RankMixer）或用户行为序列模块（例如，LONGER）已取得显著成功。然而，这些努力通常在独立的轨道上进行，这不仅阻碍了双向信息交换，还阻止了统一优化和扩展。本文提出了OneTrans，一个统一的Transformer骨干，能够同时进行用户行为序列建模和特征交互。OneTrans采用统一的分词器，将顺序和非顺序属性转换为单一的令牌序列。堆叠的OneTrans模块在相似的顺序令牌之间共享参数，同时为非顺序令牌分配特定的参数。通过因果注意力和跨请求的KV缓存，OneTrans实现了中间表示的预计算和缓存，显著降低了训练和推理过程中的计算成本。对工业规模数据集的实验结果表明，OneTrans在参数增加时能够高效扩展，始终优于强基线，并在在线A/B测试中实现每用户GMV提升5.68%。",
      "paper_summary": {
        "summary": "OneTrans, a joint effort by ByteDance and Nanyang Technological University, introduces a unified Transformer architecture for industrial recommender systems that simultaneously processes user-behavior sequences and diverse non-sequential features. This approach enables bidirectional information exchange, leverages LLM-style optimizations, and achieves improved recommendation quality (e.g., +1.53% CTR AUC) and efficiency in industrial deployments, demonstrating significant business lifts in online A/B tests.",
        "originalProblem": [
          "Conventional recommender systems suffer from architectural fragmentation, separating user-behavior sequence modeling from feature interaction.",
          "This fragmentation restricts bidirectional information flow between static features and dynamic sequences, limiting a holistic understanding of user preferences.",
          "Fragmented architectures hinder the application of unified Large Language Model (LLM) engineering optimizations and predictable performance scaling."
        ],
        "solution": [
          "OneTrans employs a unified Transformer backbone with a comprehensive tokenization strategy that converts both sequential (user behavior) and non-sequential (user, item, context) features into a single token sequence.",
          "It utilizes a \"Mixed Parameterization Transformer Block\" where sequential tokens share parameters, while each non-sequential token has distinct parameters for Q/K/V projections and FFNs, facilitating tailored interaction.",
          "A Pyramid Stack and LLM-style optimizations (e.g., Cross-Request KV Caching, FlashAttention-2, mixed precision) are integrated to enhance efficiency, reduce computation, and improve memory management during training and inference."
        ],
        "keyInsights": [
          "Unifying sequence modeling and feature interaction within a single causal Transformer architecture enables more reliable and compute-efficient performance improvements than scaling components independently.",
          "Specific design choices like an Auto-Split tokenizer for non-sequential features, timestamp-aware fusion for sequential data, and token-specific parameters for heterogeneous non-sequential tokens are crucial for optimal performance.",
          "Adopting LLM engineering optimizations (e.g., KV caching, FlashAttention, mixed precision) significantly improves training and serving efficiency, making complex Transformer models practical for industrial recommender systems."
        ],
        "results": [
          "OneTrans-L achieved +1.53% CTR AUC and +1.14% CVR AUC over DCNv2+DIN baselines in offline evaluations, with predictable performance gains observed when scaling model size, depth, and width.",
          "LLM optimizations, including the Pyramid Stack, Cross-Request KV Caching, FlashAttention-2, and mixed precision, led to substantial efficiency improvements (e.g., ~30-69% reduction in latency/runtime, ~30-58% reduction in memory) while maintaining performance.",
          "Online A/B tests demonstrated statistically significant business lifts (e.g., +4.35% orders/user, +5.68% GMV/user in Feeds) and improved user engagement, alongside maintaining or reducing p99 inference latency by -3.91% to -3.26%."
        ]
      },
      "image_url": "image/2510.26104v1.png",
      "universal_paper_id": "2510.26104",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 83,
          "last_7_days": 83
        },
        "public_total_votes": 13
      },
      "first_publication_date": "2025-10-30T03:30:12.000Z",
      "publication_date": "2025-10-30T03:30:12.000Z",
      "updated_at": "2025-10-31T04:25:34.806Z",
      "topics": [
        "Computer Science",
        "cs.IR"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "paper_group_id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "title": "Running VLAs at Real-time Speed",
      "abstract": "在本文中，我们展示了如何使用单个消费级GPU以30Hz的帧率和最多480Hz的轨迹频率运行pi0级别的多视角VLA。这使得之前被认为大型VLA模型无法实现的动态实时任务成为可能。为了实现这一目标，我们引入了一系列策略来消除模型推理中的开销。实际实验表明，采用我们策略的pi0策略在抓取下落的笔任务中实现了100%的成功率。基于这些结果，我们进一步提出了一个用于VLA实时机器人控制的全流式推理框架。代码可以在此https URL中获取。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 41,
          "last_7_days": 41
        },
        "public_total_votes": 7
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
      "id": "019a2c33-6b53-74e7-9272-fb16dadac30a",
      "paper_group_id": "019a2c33-6b53-74e7-9272-fb16dadac30a",
      "title": "Towards Personalized Treatment Plan: Geometrical Model-Agnostic Approach to Counterfactual Explanations",
      "abstract": "在我们文章中，我们描述了一种在高维空间中生成反事实解释的方法，该方法涉及四个步骤：将我们的数据集拟合到模型、找到决策边界、确定问题的约束条件，以及从该边界计算出最近的点（反事实解释）。我们提出了一种离散化的方法，在边界上找到多个离散点，然后识别出最近的可行反事实解释。我们后来的方法称为“用于边界近似的分段采样”（SSBA），其应用二分查找寻找决策边界点，然后搜索最近的边界点。在四个不同维度的数据集上，我们展示了我们的方法能够优于当前的反事实生成方法，距离减少幅度在 $5\\%$ 到 $50\\%$ 的 $L_2$ 范数上。我们的方法还可以通过限制对不可变和分类特征（如年龄、性别、身高以及其他相关特征，例如健康数据集中的情况）的更改来处理现实世界的约束。在运行时间方面，SSBA 算法生成决策边界点的效率比基于网格的方法快多个数量级。总的来说，我们的方法提供了一种简单有效的模型无关方法，可以计算最近的可行（即具有约束的现实）反事实解释。我们的所有结果和代码可以在此链接中找到：$\\href{this https URL}{this https URL dsin85691/SSBA\\_For\\_Counterfactuals}$",
      "paper_summary": {
        "summary": "Researchers introduced Segmented Sampling for Boundary Approximation (SSBA), a geometrical and model-agnostic method for generating counterfactual explanations. The approach efficiently identifies decision boundary points in high-dimensional spaces and rigorously incorporates real-world constraints, enabling the derivation of feasible and actionable personalized intervention strategies.",
        "originalProblem": [
          "Existing counterfactual explanation methods often compromise between model-agnosticism, computational efficiency, and finding the nearest, most realistic counterfactuals.",
          "High-dimensional datasets introduce significant computational and memory challenges for traditional grid-based approaches to identifying decision boundaries.",
          "Ensuring the feasibility and actionability of generated counterfactuals by incorporating real-world constraints, such as immutable features or specific value ranges, is complex with current loss-function based methods."
        ],
        "solution": [
          "The Segmented Sampling for Boundary Approximation (SSBA) method provides a geometrical, discretized approach to approximate the decision boundary of any black-box machine learning model.",
          "It employs a binary search algorithm along line segments connecting correctly classified points from different classes to efficiently locate decision boundary points, addressing high-dimensionality challenges.",
          "Real-world constraints, including immutable features and specific value ranges, are directly applied as filters to the generated boundary points, ensuring the feasibility of the resulting counterfactual explanations."
        ],
        "keyInsights": [
          "Approximating the decision boundary through segmented sampling offers a scalable and model-agnostic path to generating counterfactuals without needing model gradients.",
          "Directly filtering decision boundary points based on predefined constraints is more robust for ensuring counterfactual feasibility than embedding them in complex, multi-objective loss functions.",
          "Using binary search along connecting line segments between points of differing classes effectively navigates high-dimensional spaces, circumventing the exponential complexity of grid-based methods."
        ],
        "results": [
          "SSBA demonstrated superior scalability, generating up to a million boundary points for 50 features in seconds, proving feasible where grid-based methods failed due to memory limitations.",
          "In unconstrained scenarios, SSBA consistently reduced the L2 distance between the original instance and the counterfactual by 5% to 50% compared to other model-agnostic methods.",
          "For constrained scenarios on real-world datasets, SSBA produced closer 'bounded counterfactuals' to the decision boundary than both model-agnostic and gradient-based comparative methods, strictly adhering to defined feasibility constraints."
        ]
      },
      "image_url": "image/2510.22911v1.png",
      "universal_paper_id": "2510.22911",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 164,
          "last_7_days": 164
        },
        "public_total_votes": 20
      },
      "first_publication_date": "2025-10-27T01:28:57.000Z",
      "publication_date": "2025-10-27T01:28:57.000Z",
      "updated_at": "2025-10-28T19:02:38.675Z",
      "topics": [
        "ai-for-health",
        "Computer Science",
        "cs.LG",
        "explainable-ai",
        "model-interpretation",
        "optimization-methods",
        "statistical-learning",
        "Statistics",
        "stat.ML"
      ],
      "organization_info": [
        {
          "name": "University of Pennsylvania",
          "image": "images/organizations/upenn.jpeg"
        },
        {
          "name": "Drexel University",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 2,
      "github_url": "https://github.com/dsin85691/SSBA_For_Counterfactuals",
      "distance": 1
    },
    {
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我改进系统需要与环境互动以实现持续适应。我们介绍了SPICE（自我游戏语料环境），这是一个强化学习框架，其中单个模型扮演两个角色：挑战者从大型语料库中提取文档以生成多样的推理任务，而推理者则解决这些任务。通过对抗性动态，挑战者在推理者能力的边界上创造了一个自动课程，同时语料库的基础为持续改进提供了丰富、近乎取之不尽的外部信号。与现有的无基础自我游戏方法所带来的有限好处不同，SPICE在多个模型系列的数学（+8.9%）和一般推理（+9.8%）基准上取得了一致的提升。我们的分析揭示了文档基础是SPICE中的一个关键因素，使其能够持续生成越来越具有挑战性的目标并实现这些目标，从而实现持续的自我改进。",
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
        "total_votes": 8,
        "visits_count": {
          "all": 179,
          "last_7_days": 179
        },
        "public_total_votes": 29
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
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3afa-3bf1-7ad5-98f8-712f5cb8cdb8",
      "paper_group_id": "019a3afa-3bf1-7ad5-98f8-712f5cb8cdb8",
      "title": "LSM-MS2: A Foundation Model Bridging Spectral Identification and Biological Interpretation",
      "abstract": "绝大多数质谱数据仍未得到表征，导致其生物和化学信息未被充分利用。近年来，机器学习的进展开始填补这一空白，特别是在串联质谱数据的光谱识别等任务上。在此，我们介绍了LSM-MS2的最新一代，这是一个大规模深度学习基础模型，经过数百万个光谱的训练，以学习语义化学空间。LSM-MS2在光谱识别方面达到了领先水平，在识别复杂异构化合物的准确性上提高了30%，在复杂生物样本中提供了42%的更高正确识别率，并在低浓度条件下保持了稳健性。此外，LSM-MS2生成丰富的光谱嵌入，使得从最小的下游数据中实现直接的生物解释成为可能，成功区分疾病状态，并在多样的转化应用中预测临床结果。",
      "paper_summary": null,
      "image_url": "image/2510.26715v1.png",
      "universal_paper_id": "2510.26715",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 23,
          "last_7_days": 23
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-10-30T17:13:58.000Z",
      "publication_date": "2025-10-30T17:13:58.000Z",
      "updated_at": "2025-10-31T15:54:29.233Z",
      "topics": [
        "ai-for-health",
        "Computer Science",
        "cs.LG",
        "embedding-methods",
        "representation-learning",
        "self-supervised-learning",
        "transfer-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "paper_group_id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "title": "Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks",
      "abstract": "人类具备空间推理能力，使他们能够通过多模态观察（如视觉和听觉）理解空间。大型多模态推理模型通过学习感知和推理来扩展这些能力，在多种空间任务中展现出良好的表现。然而，针对这些模型的系统评估和公开基准仍然有限。在本次调查中，我们提供了对大型模型多模态空间推理任务的全面回顾，整理了多模态大型语言模型（MLLMs）的最新进展，并介绍了用于评估的开放基准。我们首先概述了普通空间推理，重点关注后训练技术、可解释性和架构。除了经典的二维任务外，我们还考察了空间关系推理、场景和布局理解，以及视觉问答和在三维空间中的定位。我们同样回顾了在具身人工智能方面的进展，包括视觉-语言导航和动作模型。此外，我们还考虑了音频和自我中心视频等新兴模态，这些通过新传感器为新型空间理解贡献了力量。我们相信这项调查为不断发展的多模态空间推理领域奠定了坚实基础，并提供了一些见解。在这个调查的更新信息、代码和开放基准的实现可以在该网址找到。",
      "paper_summary": {
        "summary": "A systematic survey provides a comprehensive review of multimodal spatial reasoning in large models, establishing a taxonomy of tasks and introducing open benchmarks for evaluation. This work synthesizes advancements across various domains, highlighting current capabilities and persistent challenges in equipping AI with robust spatial understanding.",
        "originalProblem": [
          "Large Language Models (LLMs) are primarily unimodal, limiting their understanding of the physical world and common-sense spatial concepts.",
          "Multimodal Large Language Models (MLLMs) lack a systematic review and standardized benchmarks specifically for their spatial reasoning capabilities.",
          "The absence of unified evaluation protocols hinders consistent progress and fair comparison of different MLLM architectures in spatial reasoning tasks."
        ],
        "solution": [
          "Developed a comprehensive taxonomy for multimodal spatial reasoning, categorizing tasks across 2D, 3D, embodied AI, video, and audio modalities.",
          "Conducted a structured literature review detailing methods like test-time scaling, post-training strategies, and architectural modifications for enhancing spatial understanding in MLLMs.",
          "Introduced and made publicly available open benchmarks and evaluation protocols to standardize the assessment of MLLMs' spatial reasoning performance."
        ],
        "keyInsights": [
          "Effective spatial reasoning in MLLMs often requires explicit geometric grounding and structured reasoning, going beyond simple prompting or generic architectural designs.",
          "Integrating external tools (e.g., 2D/3D perception modules) and dedicated spatial reasoning components significantly enhances MLLMs' ability to process and infer spatial relationships.",
          "MLLMs frequently rely on object co-occurrence patterns rather than genuine geometric understanding, underscoring the need for balanced cross-modal encoding and relation-aware attention."
        ],
        "results": [
          "Specialized prompting strategies leveraging visual cues or structured textual prompts prove more effective for spatial tasks compared to generic Chain-of-Thought approaches.",
          "Incorporating explicit spatial cues into MLLM inputs (e.g., marker channels, depth maps) or employing dedicated architectural modifications improves spatial structure preservation.",
          "While benchmarks are evolving, current MLLM evaluations for spatial reasoning still face limitations in generalization to real-world dynamic scenes, cross-modal alignment, and annotation scalability."
        ]
      },
      "image_url": "image/2510.25760v1.png",
      "universal_paper_id": "2510.25760",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 75,
          "last_7_days": 75
        },
        "public_total_votes": 15
      },
      "first_publication_date": "2025-10-29T17:55:43.000Z",
      "publication_date": "2025-10-29T17:55:43.000Z",
      "updated_at": "2025-10-30T05:59:27.668Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "explainable-ai",
        "multi-modal-learning",
        "object-detection",
        "reasoning",
        "robotics-perception",
        "transformers",
        "vision-language-models",
        "visual-qa"
      ],
      "organization_info": [
        {
          "name": "South China University of Technology",
          "image": null
        },
        {
          "name": "Shanghai Jiao Tong University",
          "image": "images/organizations/shanghai-jiao-tong-university.png"
        },
        {
          "name": "University of Pisa",
          "image": null
        },
        {
          "name": "HKUST(GZ)",
          "image": null
        },
        {
          "name": "INSAIT",
          "image": null
        },
        {
          "name": "HKUST",
          "image": "images/organizations/hkust.jpg"
        },
        {
          "name": "University of Trento",
          "image": null
        },
        {
          "name": "Sofia University \"St. Kliment Ohridski\"",
          "image": null
        },
        {
          "name": "Sofia University “St. Kliment Ohridski”",
          "image": null
        },
        {
          "name": "Sofia University \",St. Kliment Ohridski\"",
          "image": null
        },
        {
          "name": "Sofia University \",\"St. Kliment Ohridski\"",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 18,
      "github_url": "https://github.com/zhengxuJosh/Awesome-Multimodal-Spatial-Reasoning",
      "distance": 1
    },
    {
      "id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "paper_group_id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "title": "Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark",
      "abstract": "最近的视频生成模型能够生成高保真、时间一致的视频，这表明它们可能编码了大量的世界知识。除了现实的合成外，它们还表现出新的行为，表明其具备视觉感知、建模和操控能力。然而，一个重要的问题仍然存在：视频模型是否准备好在具有挑战性的视觉推理场景中充当零-shot 推理器？在本研究中，我们进行了实证研究，以全面探讨这个问题，重点关注领先的流行模型Veo-3。我们评估其在空间、几何、物理、时间和具身逻辑等12个维度上的推理行为，系统地表征其优点和失效模式。为了标准化这项研究，我们将评估数据整理成MME-CoF，一个紧凑的基准，能够深入和全面地评估链帧（CoF）推理。我们的发现揭示，尽管当前的视频模型在短期空间一致性、细粒度基础和局部一致动态方面展现出有希望的推理模式，但在长期因果推理、严格几何约束和抽象逻辑方面仍然有限。总体来说，它们尚不能作为独立的零-shot 推理器，但作为专门推理模型的补充视觉引擎展现出令人鼓舞的迹象。项目页面：此网址。",
      "paper_summary": {
        "summary": "This study empirically investigates the zero-shot reasoning capabilities of state-of-the-art video generation models using a new benchmark, MME-COF, finding that current models are not yet ready as standalone reasoners but exhibit encouraging signs as complementary visual engines. It reveals that their strong generative performance often stems from learning surface-level patterns rather than a deep, principle-driven understanding of visual logic.",
        "originalProblem": [
          "Unclear if advanced video generation models, beyond producing high-fidelity content, possess genuine reasoning capabilities in a zero-shot setting.",
          "A lack of standardized benchmarks specifically designed to rigorously evaluate sequential visual reasoning (Chain-of-Frame) in generative video models across diverse cognitive dimensions.",
          "Distinguishing whether strong generative performance reflects an internalization of general principles or merely learning surface-level patterns from training data."
        ],
        "solution": [
          "Developed MME-COF, a new benchmark comprising 12 diverse reasoning categories (e.g., physics, geometry, medical, embodied) with over 100 cases, meticulously curated by PhD-level experts.",
          "Established a rigorous and unified prompt design protocol to ensure consistency, minimize linguistic bias, and explicitly test zero-shot Chain-of-Frame reasoning.",
          "Evaluated leading video models (e.g., Veo-3, Sora-2, Kling, Seedance) using human qualitative judgments and automated scoring via Gemini-2.5-Pro on generated video samples."
        ],
        "keyInsights": [
          "Current video models largely exhibit pattern-driven behavior, demonstrating proficiency in short-term visual coherence but struggling with long-horizon causality, abstract logical principles, and adherence to quantitative constraints.",
          "The 'Chain-of-Frame' reasoning concept shows promise, as models exhibit emergent abilities in short-horizon trace consistency and simulating basic tool-use, hinting at potential for iterative visual problem-solving.",
          "While not yet reliable standalone reasoners, video models show potential as 'complementary visual engines' that could be integrated with dedicated reasoning models for next-generation collaborative visual AI systems."
        ],
        "results": [
          "Most evaluated models exhibited limited reasoning capability across MME-COF tasks, with average scores below 2.0 out of 4.0 (Gemini-2.5-Pro), despite achieving higher scores in visual stability.",
          "Models consistently failed in tasks requiring complex causal and physical logic, long-horizon rule-grounded reasoning, precise geometric transformations, and functional understanding in GUI or medical contexts.",
          "Observed strengths included reasonable performance in fine-grained visual detail grounding for salient targets, producing locally coherent short-horizon trace animations, and preliminary grasping of simple 3D geometry and real-world spatial layouts."
        ]
      },
      "image_url": "image/2510.26802v1.png",
      "universal_paper_id": "2510.26802",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 36,
          "last_7_days": 36
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-30T17:59:55.000Z",
      "publication_date": "2025-10-30T17:59:55.000Z",
      "updated_at": "2025-10-31T02:52:10.188Z",
      "topics": [
        "causal-inference",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "generative-models",
        "reasoning",
        "video-understanding",
        "visual-reasoning",
        "zero-shot-learning"
      ],
      "organization_info": [
        {
          "name": "Northeastern University",
          "image": "images/organizations/northeastern.png"
        },
        {
          "name": "CUHK",
          "image": "images/organizations/chinesehongkong.png"
        },
        {
          "name": "Peking University",
          "image": "images/organizations/peking.png"
        },
        {
          "name": "MMLab",
          "image": null
        },
        {
          "name": "IMIXR",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/OpenGVLab/LLaMA-Adapter",
      "distance": 1
    },
    {
      "id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "paper_group_id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "title": "GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning",
      "abstract": "基于大语言模型（LLMs）的自主代理在复杂任务解决中的工具操作能力令人印象深刻。然而，现有的诸如ReAct的范式依赖于顺序推理和执行，未能充分利用独立子任务之间的内在并行性。这种顺序瓶颈导致工具利用效率低下和多步骤推理场景下的性能不佳。我们提出了基于图的代理规划（GAP），这是一个新颖的框架，通过基于图的规划明确建模任务间的依赖关系，从而实现自适应的并行和串行工具执行。我们的方法训练代理基础模型，将复杂任务分解为考虑依赖关系的子任务图，自动确定哪些工具可以并行执行，哪些工具必须遵循顺序依赖。这种考虑依赖关系的协调在执行效率和任务准确性上都实现了显著提升。为了训练GAP，我们构建了一个高质量的基于图的规划轨迹数据集，该数据集源自多跳问答（MHQA）基准。我们采用了两阶段的训练策略：在精心挑选的数据集上进行监督微调（SFT），然后在经过战略性抽样的查询上进行带有基于正确性的奖励函数的强化学习（RL），这些查询的工具推理提供最大价值。MHQA数据集上的实验结果表明，GAP显著优于传统的ReAct基线，尤其是在多步骤检索任务上，同时通过智能并行化实现了工具调用效率的剧烈提升。项目页面可在此 https URL 查看。",
      "paper_summary": {
        "summary": "The Graph-based Agent Planning (GAP) framework empowers large language model agents to construct and reason over graph-based task dependencies, facilitating parallel tool execution. This methodology significantly reduces LLM interaction turns by up to 33.8% and execution time by 32.3%, while also improving accuracy on complex multi-hop reasoning tasks.",
        "originalProblem": [
          "Existing tool-augmented LLM agents, like ReAct, are limited by a sequential 'think-act-observe' execution model, preventing parallel tool use for independent sub-tasks.",
          "The sequential bottleneck leads to inefficient tool utilization, increased waiting times, and higher computational overhead for complex multi-step reasoning.",
          "Current approaches often rely on brittle prompt engineering or incur high communication costs in multi-agent systems, lacking inherent LLM capabilities for optimal coordination."
        ],
        "solution": [
          "GAP introduces a graph-based planning paradigm where the LLM explicitly constructs a task dependency graph for sub-tasks, identifying opportunities for parallel execution.",
          "A dependency-aware execution strategy is employed, topologically sorting the graph into levels, allowing all independent sub-tasks within a level to be executed in parallel.",
          "The agent undergoes a two-stage training pipeline: Supervised Fine-Tuning (SFT) on synthesized graph-based planning traces, followed by Reinforcement Learning (RL) to optimize for efficiency and correctness."
        ],
        "keyInsights": [
          "Explicitly training LLMs to model and reason about task dependencies using a graph representation allows for a more efficient and human-like planning capability.",
          "Parallel execution of independent tool calls within a single agent significantly reduces interaction turns, execution time, and overall computational cost.",
          "A combination of supervised learning to establish foundational planning and reinforcement learning to optimize execution strategy yields superior performance and cost-effectiveness for complex tasks."
        ],
        "results": [
          "GAP-3B achieved an average accuracy improvement of 0.9% on multi-hop QA datasets compared to the best baseline (AFM-RL-3B).",
          "It reduced LLM interaction turns by 21.6% (HotpotQA) to 33.4% (2WikiMultiHopQA) and execution time by 32.3% (HotpotQA) to 21.4% (2WikiMultiHopQA) compared to baselines.",
          "The framework lowered response length by 24.9% on HotpotQA, demonstrating a more efficient reasoning process and a superior performance-cost trade-off across various benchmarks."
        ]
      },
      "image_url": "image/2510.25320v1.png",
      "universal_paper_id": "2510.25320",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 46,
          "last_7_days": 46
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-29T09:35:55.000Z",
      "publication_date": "2025-10-29T09:35:55.000Z",
      "updated_at": "2025-10-30T11:04:33.057Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "fine-tuning",
        "reasoning",
        "reinforcement-learning",
        "tool-use",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a307c-9cab-7643-91a6-f1546f864a07",
      "paper_group_id": "019a307c-9cab-7643-91a6-f1546f864a07",
      "title": "ParallelMuse: Agentic Parallel Thinking for Deep Information Seeking",
      "abstract": "平行思维扩展了探索的广度，补充了信息获取（IS）代理的深度探索，从而进一步增强了问题解决能力。然而，在这种设置中，传统的平行思维面临两个关键挑战：一是反复从头开始的低效率，二是在生成答案时整合长时间推理轨迹的困难，因为有限的上下文容量无法充分考虑推理过程。为了解决这些问题，我们提出了ParallelMuse，一种为深度IS代理设计的两阶段范式。第一阶段，功能指定的部分展开，将生成的序列划分为功能区域，并进行不确定性引导的路径重用和分支，以提高探索效率。第二阶段，压缩推理聚合，利用推理冗余无损压缩与答案推导相关的信息，并综合出一个连贯的最终答案。在多个开源代理和基准测试中的实验表明，性能提高最高可达62%，同时探索性令牌的消耗减少10%至30%。",
      "paper_summary": {
        "summary": "Alibaba Group's Tongyi Lab introduced PARALLELMUSE, a two-stage parallel thinking paradigm designed to optimize deep information-seeking agents. This method enhances problem-solving capability and efficiency by leveraging targeted exploration and lossless reasoning compression, achieving up to 62% performance improvement on benchmarks while providing substantial token and context savings.",
        "originalProblem": [
          "Existing parallel thinking methods are inefficient for deep information-seeking agents due to redundant computations and treating all agent behaviors homogeneously.",
          "Aggregating long, complex agentic reasoning trajectories is challenging within large language model (LLM) context limits, often forcing the discarding of crucial intermediate steps.",
          "LLM self-confidence scores are often unreliable for answer selection in agentic tasks, as external tool responses can distort internal probability distributions."
        ],
        "solution": [
          "Implemented a Functionality-Specified Partial Rollout (Stage 1) that intelligently identifies high-uncertainty branching points in distinct functional regions (reasoning or exploration) and reuses context via KV cache for efficient sampling.",
          "Developed Compressed Reasoning Aggregation (Stage 2) where each reasoning trajectory is condensed into a structured, lossless report that preserves only essential information for answer derivation.",
          "Utilized an aggregator LLM to synthesize a final answer from these compressed reports, prioritizing reasoning coherence over mere answer consistency without further tool invocations."
        ],
        "keyInsights": [
          "Deep information-seeking agent trajectories contain distinct functional regions (reasoning vs. tool-use) that exhibit different uncertainty patterns, allowing for more targeted and efficient exploration.",
          "Agentic reasoning trajectories often have high redundancy, indicating potential for significant lossless compression to integrate more intermediate information within LLM context windows.",
          "The integration of external, non-model-generated tool responses can lead to miscalibration of LLM confidence scores, making confidence-based aggregation strategies less reliable for agentic tasks."
        ],
        "results": [
          "Achieved up to 62% performance improvement across various open-source deep IS agents and benchmarks (e.g., BrowseComp, GAIA), consistently outperforming all baselines.",
          "Demonstrated significant efficiency gains, including up to 28% token savings from functionality-specified partial rollouts and up to 99% context token reduction through compressed reasoning aggregation.",
          "Exhibited robustness against confidence miscalibration, leading to more consistent improvements than confidence-based methods and achieving performance comparable to or surpassing many closed-source deep IS agents."
        ]
      },
      "image_url": "image/2510.24698v1.png",
      "universal_paper_id": "2510.24698",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 55,
          "last_7_days": 55
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-28T17:51:50.000Z",
      "publication_date": "2025-10-28T17:51:50.000Z",
      "updated_at": "2025-10-29T15:01:04.299Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "inference-optimization",
        "information-extraction",
        "model-compression",
        "reasoning",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/Alibaba-NLP/DeepResearch",
      "distance": 1
    },
    {
      "id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "paper_group_id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "title": "Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model",
      "abstract": "最近的大型语言模型（LLM）研究经历了从编码-解码建模到如今主流的仅解码建模的架构转变。然而，这一快速转变缺乏严格的比较分析，特别是\\textit{从扩展的角度}来看，引发了对编码-解码模型潜力可能被忽视的担忧。为填补这一空白，我们重新审视了编码-解码LLM（RedLLM），并结合了近期仅解码LLM（DecLLM）的新策略。我们在不同模型规模下进行全面比较，规模从$\\sim$150M到$\\sim$8B不等，比较了使用前缀语言建模（LM）进行预训练的RedLLM与使用因果LM进行预训练的DecLLM。利用RedPajama V1（1.6T tokens）进行预训练，并使用FLAN进行指令调优，我们的实验表明RedLLM展现出引人注目的扩展特性和令人惊讶的强大性能。尽管DecLLM在预训练过程中总体上更具计算优化性，RedLLM则展现出可比的扩展性和上下文长度外推能力。经过指令调优后，RedLLM在各种下游任务上实现了可比甚至更好的结果，同时享受了显著更好的推理效率。我们希望我们的发现能够激励更多对RedLLM的重新审视，从而释放其开发强大而高效的LLM的潜力。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 18,
          "last_7_days": 18
        },
        "public_total_votes": 5
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
      "id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "paper_group_id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "title": "DeepSeek-OCR: Contexts Optical Compression",
      "abstract": "我们提出了DeepSeek-OCR，作为对通过光学二维映射压缩长文本上下文可行性的初步研究。DeepSeek-OCR由两个组件组成：DeepEncoder和DeepSeek3B-MoE-A570M作为解码器。具体而言，DeepEncoder作为核心引擎，旨在在高分辨率输入下保持低激活，同时实现高压缩比，以确保视觉标记数量的最佳和可管理。实验表明，当文本标记数量在视觉标记的10倍以内（即压缩比< 10x）时，模型的解码（OCR）精度可达97%。即便在20倍的压缩比下，OCR准确率仍保持在约60%。这为历史长文本压缩和大规模语言模型中的记忆遗忘机制等研究领域显示出巨大的潜力。此外，DeepSeek-OCR也展现出了较高的实用价值。在OmniDocBench上，它仅使用100个视觉标记就超过了GOT-OCR2.0（256个标记/页），并在使用不足800个视觉标记的情况下，优于MinerU2.0（平均每页6000多个标记）。在生产中，DeepSeek-OCR可以以每天超过20万页的规模为大规模语言模型/视觉语言模型生成训练数据（一个A100-40G）。代码和模型权重可以在这个http URL上公开访问。",
      "paper_summary": {
        "summary": "DeepSeek-OCR explores \"contexts optical compression\" to enable Large Language Models (LLMs) to process lengthy texts more efficiently by representing information visually. The model achieves approximately 97% text decoding precision at 9-10x vision-text compression ratios and sets new benchmarks in OCR performance with significantly fewer vision tokens, while also offering advanced deep parsing capabilities for structured data.",
        "originalProblem": [
          "Large Language Models (LLMs) face significant computational and memory challenges due to quadratic scaling with increasing input sequence length, limiting their ability to process long textual contexts.",
          "Existing vision encoders in Vision-Language Models (VLMs) suffer from limitations such as complex deployment, excessive image fragmentation, or high activation memory consumption for high-resolution documents.",
          "Traditional OCR models have not explicitly addressed the optimal vision-text compression ratio or the minimum number of vision tokens required for accurate text decoding when integrating with LLMs for long-context understanding."
        ],
        "solution": [
          "DeepSeek-OCR is an end-to-end Vision-Language Model consisting of a novel 380M-parameter DeepEncoder and a DeepSeek3B-MoE decoder.",
          "The DeepEncoder combines SAM-base and CLIP-large models with a crucial 16x Token Compressor (a 2-layer convolutional network) to process high-resolution images efficiently, reduce activation memory, and generate a minimal number of vision tokens.",
          "The model supports multiple resolution modes, including dynamic 'Gundam' modes, to handle ultra-high-resolution inputs via tiling, and is trained on a diverse dataset encompassing traditional OCR, complex artificial images (charts, formulas, geometry), and general vision data."
        ],
        "keyInsights": [
          "The visual modality can serve as an effective and quantifiable compression layer for textual information, allowing an image to represent rich text with substantially fewer tokens than its digital counterpart, thereby mitigating LLM long-context issues.",
          "Balancing local visual detail perception (SAM) with global contextual understanding (CLIP) while drastically reducing token count via an intermediate compressor enables high-resolution input processing with minimal vision token overhead.",
          "The concept of progressively downsizing rendered images to simulate 'optical forgetting' suggests a biologically inspired mechanism for creating theoretically unlimited context architectures in LLMs."
        ],
        "results": [
          "The model achieves approximately 97% decoding precision on text-rich documents with 600-1300 text tokens when the vision-text compression ratio is within 10x, with precision remaining around 60% even at 20x compression.",
          "DeepSeek-OCR, using 100 vision tokens, outperforms existing models like GOT-OCR2.0 (256 tokens) on the OmniDocBench and surpasses MinerU2.0 (nearly 7,000 tokens) in its Gundam mode (fewer than 800 tokens).",
          "It demonstrates advanced 'deep parsing' capabilities, extracting structured information from charts (to HTML), chemical formulas (to SMILES), and geometric figures, supporting nearly 100 languages, and generates over 200,000 pages per day on a single A100 GPU for training data."
        ]
      },
      "image_url": "image/2510.18234v1.png",
      "universal_paper_id": "2510.18234",
      "metrics": {
        "total_votes": 257,
        "visits_count": {
          "all": 8514,
          "last_7_days": 4805
        },
        "public_total_votes": 496
      },
      "first_publication_date": "2025-10-21T02:41:44.000Z",
      "publication_date": "2025-10-21T02:41:44.000Z",
      "updated_at": "2025-10-22T02:59:27.222Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "efficient-transformers",
        "inference-optimization",
        "information-extraction",
        "multi-modal-learning",
        "synthetic-data",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "DeepSeek",
          "image": "images/organizations/deepseek.png"
        }
      ],
      "author_info": [],
      "github_stars": 9072,
      "github_url": "https://github.com/deepseek-ai/DeepSeek-OCR",
      "distance": 1
    },
    {
      "id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "paper_group_id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "title": "Parallel Loop Transformer for Efficient Test-Time Computation Scaling",
      "abstract": "大型语言模型（LLMs）功能强大，但在推理过程中通常太慢且成本过高，不适合实际应用。循环变压器通过在多个计算步骤或“循环”中重复使用相同的权重来节省参数。然而，这种方法有一个主要缺陷：循环依次运行，导致每增加一个循环，推理延迟和内存需求都会增加。这使得它们在快速应用中不切实际。为了解决这个问题，我们引入了并行循环变压器（PLT）。PLT是一种新架构，能够提供深层循环模型的性能优势，同时具备标准非循环模型的低延迟。PLT通过两种关键技术工作。首先，跨循环并行性（CLP）通过同时计算不同令牌的不同循环，打破了顺序依赖，所有操作都在一次传递中完成。其次，为了防止内存成本增长，我们采用了一种高效表示增强策略。该方法将第一个循环的内存（KV缓存）与所有其他循环共享。然后，它使用门控滑动窗口注意力（G-SWA）将这些共享的全局信息与局部信息结合在一起，保持高准确度。我们的实验表明，PLT在与标准变压器相比几乎没有额外延迟或内存成本的情况下，达到了传统循环模型的高准确度。",
      "paper_summary": {
        "summary": "The Parallel Loop Transformer (PLT) from ByteDance Seed allows Large Language Models to leverage deep, looped computation for enhanced reasoning without incurring the traditional inference latency and memory overhead. This architecture achieves accuracy comparable to or exceeding larger vanilla models with up to 30% lower latency and reduced Key-Value cache memory.",
        "originalProblem": [
          "Large Language Models (LLMs) suffer from high inference latency and memory consumption, limiting their deployment in real-time or resource-constrained applications.",
          "Existing looped transformer architectures, despite parameter efficiency, execute computational loops sequentially, causing per-token latency and KV-cache memory to scale linearly with the number of loops.",
          "This inherent sequential dependency makes deeper, reasoning-capable looped models impractical due to prohibitive inference costs."
        ],
        "solution": [
          "The Parallel Loop Transformer (PLT) employs Cross-Loop Parallelism (CLP) to reconfigure training dependencies, enabling the concurrent execution of multiple loop steps during inference.",
          "It implements an Efficient Representation Enhancement strategy that shares the Key-Value (KV) cache from the first loop across all subsequent loops, drastically reducing memory footprint.",
          "Non-first loops utilize Gated Sliding-Window Attention (G-SWA) to restore local context, adaptively fusing global shared information with a small, local sliding window."
        ],
        "keyInsights": [
          "The inherent sequential bottleneck of looped transformers can be overcome by carefully shifting dependencies during training, enabling full parallelization of loop computation during inference.",
          "Decoupling an LLM's effective computational depth and reasoning capability from its inference-time latency and memory footprint is feasible through innovative architectural design.",
          "Efficiently managing memory in deep looped models can be achieved by sharing global KV-cache components while restoring fine-grained local context through gated sliding-window attention."
        ],
        "results": [
          "PLT with two loops (PLT-2) matched the accuracy of a naive looped transformer while achieving latency within 2% and KV cache within 1.4% of a vanilla non-looped transformer.",
          "A 1.7B parameter PLT-2 model outperformed a larger 2.5B parameter vanilla model in accuracy (62.6 vs 62.1), simultaneously reducing decoding latency by approximately 30% and significantly decreasing KV cache memory.",
          "Scaling PLT to three loops (PLT-3) further improved average accuracy by 1.1 points with only marginal increases in latency (+2%) and KV cache (+1.1%), showcasing its efficient scalability."
        ]
      },
      "image_url": "image/2510.24824v1.png",
      "universal_paper_id": "2510.24824",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 77,
          "last_7_days": 77
        },
        "public_total_votes": 16
      },
      "first_publication_date": "2025-10-28T15:35:50.000Z",
      "publication_date": "2025-10-28T15:35:50.000Z",
      "updated_at": "2025-10-30T03:16:39.208Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.CL",
        "efficient-transformers",
        "inference-optimization",
        "lightweight-models",
        "model-compression",
        "parameter-efficient-training",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "ByteDance",
          "image": "images/organizations/bytedance.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a38b4-0bfd-7cdf-80c2-26e35fe72400",
      "paper_group_id": "019a38b4-0bfd-7cdf-80c2-26e35fe72400",
      "title": "Remote Labor Index: Measuring AI Automation of Remote Work",
      "abstract": "人工智能在以知识和推理为导向的研究基准上取得了快速进展，但这些进展如何转化为经济价值和自动化仍不明确。为了衡量这一点，我们引入了远程劳动指数（RLI），这是一个覆盖多个行业的基准，其中包含旨在评估端到端代理在实际环境中表现的现实经济价值项目。AI代理在RLI上的表现接近底线，表现最好的代理实现了2.5%的自动化率。这些结果有助于通过实证证据来为人工智能自动化的讨论提供基础，建立一个共同的基础，以跟踪人工智能的影响，并使利益相关者能够主动应对人工智能驱动的劳动自动化。",
      "paper_summary": {
        "summary": "The Remote Labor Index (RLI) establishes a new empirical benchmark to evaluate AI agents on their ability to perform economically valuable, real-world remote tasks sourced from freelance platforms. Initial evaluations show frontier AI models automate less than 3% of these end-to-end projects, though relative performance metrics indicate gradual improvements.",
        "originalProblem": [
          "Existing AI benchmarks primarily focus on academic knowledge or isolated technical skills, failing to directly measure AI's capacity for economically valuable, end-to-end project completion.",
          "There is a lack of standardized, empirical methods to quantify AI's automation potential in real-world remote work, leading to speculative discussions and an inability to track progress consistently.",
          "Prior evaluations often use simplified environments or focus on narrow task domains, not capturing the diverse, multi-modal complexity of the broader remote labor market."
        ],
        "solution": [
          "The Remote Labor Index (RLI) curates 240 end-to-end remote freelance projects directly from platforms like Upwork, ensuring tasks are economically valuable and reflect real-world demands.",
          "It covers 23 diverse job categories (e.g., graphic design, audio production, game development, architecture) and numerous file formats, addressing the multi-modal complexity and breadth of remote work.",
          "RLI employs rigorous manual evaluation by trained experts using metrics such as automation rate, Elo score, and economic value (dollars earned, autoflation) to assess AI agent performance comprehensively."
        ],
        "keyInsights": [
          "Current frontier AI agents achieve very low absolute automation rates on complex, end-to-end remote tasks, indicating a significant gap between present capabilities and full project completion.",
          "Qualitative analysis reveals common AI failure modes including technical/file integrity issues, incomplete deliverables, poor quality, and inconsistencies, highlighting areas for future research.",
          "The RLI's Elo score effectively measures incremental progress and differentiates between models, demonstrating its sensitivity for tracking AI advancements even when full automation is not yet achieved."
        ],
        "results": [
          "The highest-performing AI agent, Manus, automated only 2.5% of RLI projects, with other leading models (e.g., Grok 4, Sonnet 4.5, GPT-5) achieving automation rates between 0.8% and 2.1%.",
          "AI agents scored in the 400s and 500s on the Elo scale, significantly below the human baseline of 1,000, underscoring substantial room for improvement in relative performance.",
          "Analysis of AI failures indicated poor quality deliverables (45.6%), incomplete or malformed outputs (35.7%), and technical/file integrity issues (17.6%) as prevalent shortcomings."
        ]
      },
      "image_url": "image/2510.26787v1.png",
      "universal_paper_id": "2510.26787",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 17,
          "last_7_days": 17
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-10-30T17:58:04.000Z",
      "publication_date": "2025-10-30T17:58:04.000Z",
      "updated_at": "2025-10-31T05:18:35.005Z",
      "topics": [
        "agent-based-systems",
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "industrial-automation",
        "ml-systems",
        "reasoning",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3b2c-07e6-747c-91f3-db8bd8b1b117",
      "paper_group_id": "019a3b2c-07e6-747c-91f3-db8bd8b1b117",
      "title": "POWSM: A Phonetic Open Whisper-Style Speech Foundation Model",
      "abstract": "最近在口语语言处理方面的进展导致了语音任务的重大进展，如自动语音识别（ASR）、音素识别（PR）、图形到音素转换（G2P）和音素到图形转换（P2G）。尽管这些任务在概念上相似，但它们在很大程度上是孤立研究的，各自依赖于特定于任务的架构和数据集。在本文中，我们介绍了POWSM（音素开放风格语音模型），这是第一个能够联合执行多个与音素相关任务的统一框架。POWSM实现了音频、文本（图形）和音素之间的无缝转换，为通用和低资源语音处理开辟了新的可能性。我们的模型超越或匹配与之大小相似的专用PR模型（Wav2Vec2Phoneme和ZIPA），同时支持G2P、P2G和ASR。我们发布了训练数据、代码和模型，以促进开放科学。",
      "paper_summary": {
        "summary": "POWSM is an open-source, whisper-style speech foundation model that unifies automatic speech recognition, phone recognition, and audio-guided grapheme-to-phoneme and phoneme-to-grapheme conversion within a single architecture. The model achieves the lowest average Phonetic Feature Error Rate (2.62) on the in-domain IPAPack++ dataset and demonstrates strong generalization to unseen languages and low-resource ASR.",
        "originalProblem": [
          "Speech processing tasks like ASR, PR, G2P, and P2G have traditionally been developed in isolation, leading to fragmented systems.",
          "Existing large speech foundation models often handle phonetic information implicitly, without explicitly outputting or operating on phonemes.",
          "The absence of a unified, explicit phonetic framework hinders universal speech processing and effective support for low-resource languages."
        ],
        "solution": [
          "POWSM employs a multi-task learning approach within an attention-based encoder-decoder (AED) architecture, inspired by OWSM.",
          "It is trained on IPAPack++ (approximately 17,000 hours of multilingual speech with orthographic and phonemic transcriptions) and incorporates refined English G2P sequences for phonetic accuracy.",
          "Standard ASR datasets are reformulated into four task-specific formats (PR, ASR, audio-guided G2P, audio-guided P2G), each using a text prompt, language token, and task token to enable joint learning."
        ],
        "keyInsights": [
          "POWSM's multi-task training and decoder's language modeling capabilities effectively unify disparate phonetic tasks, achieving superior in-domain phone recognition.",
          "Explicit phonetic pre-training significantly benefits low-resource ASR, particularly when using predicted phones as prompts for P2G tasks.",
          "The encoder-CTC component of POWSM shows a preference for fine-grained phone tokens without suprasegmentals, suggesting that simpler, atomic phonetic units are more effective for the encoder's alignment task."
        ],
        "results": [
          "POWSM achieved the lowest average Phonetic Feature Error Rate (PFER) of 2.62 on the in-domain IPAPack++ dataset for phone recognition, outperforming several specialized PR models.",
          "The model demonstrated strong generalization to out-of-domain and unseen languages (e.g., DoReCo, VoxAngeles), achieving an average PFER of 18.71.",
          "POWSM improved ASR performance for low-resource languages, with the PR-P2G approach significantly reducing Word Error Rate compared to similar-sized multilingual foundation models."
        ]
      },
      "image_url": "image/2510.24992v1.png",
      "universal_paper_id": "2510.24992",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 11,
          "last_7_days": 11
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-10-28T21:43:45.000Z",
      "publication_date": "2025-10-28T21:43:45.000Z",
      "updated_at": "2025-10-31T16:48:52.710Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "fine-tuning",
        "multi-modal-learning",
        "multi-task-learning",
        "representation-learning",
        "sequence-modeling",
        "speech-recognition",
        "transfer-learning",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3821-2691-78b1-9b87-f58b6be0bba6",
      "paper_group_id": "019a3821-2691-78b1-9b87-f58b6be0bba6",
      "title": "Rapid Brightening of 3I/ATLAS Ahead of Perihelion",
      "abstract": "星际彗星3I/ATLAS正在接近其2025年10月29日的近日点，此时它位于地球与太阳的相对位置，使得过去一个月的地面光学观测受到阻碍。然而，这种几何位置使得这颗彗星进入了几个空间太阳日冕仪和太阳圈成像仪的视野中，使得它在接近近日点的过程中得以继续观察。我们报告了来自STEREO-A的SECCHI HI1和COR2、SOHO的LASCO C3以及GOES-19的CCOR-1仪器在2025年9月至10月期间的光度测量结果，这些结果显示彗星的亮度随着日心距离r的变化而迅速上升，呈现为r^(-7.5+/-1.0)。CCOR-1还将彗星解析为一个扩展源，表面彗发的视直径约为4'。此外，LASCO的颜色光度测量表明，彗星的颜色明显比太阳更蓝，符合在近日点附近气体发射对可见亮度贡献了相当大的份额。",
      "paper_summary": {
        "summary": "The activity of interstellar comet 3I/ATLAS during its approach to perihelion, unobservable from Earth due to solar conjunction, was characterized using space-based solar observatories. The study revealed an exceptionally rapid brightening rate of r⁻⁷·⁵±¹·⁰ and a distinctly blue color, indicating dominant gas-driven activity, likely from H₂O sublimation, at its closest approach to the Sun.",
        "originalProblem": [
          "Ground-based optical observations of interstellar comet 3I/ATLAS were impossible for a crucial month leading up to perihelion due to its superior conjunction.",
          "Understanding the physical mechanisms driving the comet's previously observed rapid brightening and distinguishing between dust and gas contributions during its peak solar heating phase.",
          "Characterizing 3I/ATLAS's activity near perihelion is crucial for deciphering its volatile inventory, internal structure, and interstellar origin, which was hindered by the observational gap."
        ],
        "solution": [
          "Leveraged data from multiple space-based solar observatories (STEREO-A HI1/COR2, SOHO LASCO C3, GOES-19 CCOR-1) to track 3I/ATLAS during its solar conjunction period.",
          "Applied a standardized data processing pipeline, including image stacking to enhance signal-to-noise, and photometric calibration to derive time-series brightness measurements.",
          "Utilized color photometry from SOHO LASCO C3's filters and GOES-19 CCOR-1's bluer bandpass to differentiate between dust and gas contributions to the comet's visible light."
        ],
        "keyInsights": [
          "The strong correlation between the optical brightening rate and OH production indicates that H₂O sublimation is the primary driver of 3I/ATLAS's intense activity near the Sun.",
          "The exceptionally rapid, gas-driven brightening and blue color observed distinguish 3I/ATLAS from 2I/Borisov, suggesting diverse compositions or evolutionary histories among interstellar objects.",
          "The successful and robust use of space-based solar observatories for cometary photometry demonstrates a powerful, opportunistic approach to astronomy, maximizing scientific return from existing missions."
        ],
        "results": [
          "3I/ATLAS was resolved as an extended source with an apparent coma approximately 4 arcminutes in diameter, indicative of substantial volatile release.",
          "The comet exhibited an exceptionally rapid brightening rate of r⁻⁷·⁵±¹·⁰ during its final approach to perihelion (r ≲ 2 au), significantly steeper than previously observed rates.",
          "Color photometry revealed 3I/ATLAS to be distinctly bluer than the Sun, consistent with dominant gas emission (e.g., C₂ and NH₂ bands) contributing to its visible brightness.",
          "The derived optical brightening rate closely matched independent OH radio emission rates (n = 8.3 ± 0.6), implying that H₂O sublimation largely drives the rapid optical brightening."
        ]
      },
      "image_url": "image/2510.25035v1.png",
      "universal_paper_id": "2510.25035",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 21,
          "last_7_days": 21
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-28T23:31:00.000Z",
      "publication_date": "2025-10-28T23:31:00.000Z",
      "updated_at": "2025-10-31T02:38:08.018Z",
      "topics": [
        "astro-ph.EP",
        "astro-ph.GA",
        "Physics"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "paper_group_id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "title": "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation",
      "abstract": "大型语言模型（LLMs）的近期成功重新引发了人们对推荐系统是否能够实现类似规模效益的兴趣。传统的推荐系统，主要由庞大的嵌入表构成，随着嵌入维度的增加往往会呈现饱和状态。相比之下，新的生成范式用自回归Transformer生成的紧凑语义ID（SID）序列替代了嵌入。然而，大多数工业应用仍然是专有的，留下两个基本问题未得到解决：（1）预期的规模法则在公共基准测试上是否成立？（2）是什么样的最小后训练方案可以实现竞争性能？\n\n我们推出了MiniOneRec，据我们所知，这是第一个完全开源的生成推荐框架，提供了覆盖SID构建、监督微调和面向推荐的强化学习的端到端工作流程。我们通过残差量化变分自编码器生成SID，并对范围从0.5B到7B参数的Qwen骨干网络在亚马逊评论数据集上进行了后训练。我们的实验结果揭示，随着模型规模增大，训练和评估损失均呈现出一致的下降趋势，验证了生成方法的参数效率。为了进一步提升性能，我们提出了一种轻量且有效的后训练流程，(1) 强制全流程SID对齐，(2) 应用带约束解码和混合奖励的强化学习。这些技术结合起来在排名准确性和候选多样性上都取得了显著改进。",
      "paper_summary": {
        "summary": "An open-source framework, MiniOneRec, validates large language model-like scaling laws for generative recommendation on public benchmarks while providing an efficient post-training recipe that achieves superior performance and transferability over existing methods.",
        "originalProblem": [
          "Traditional embedding-centric recommender systems exhibit performance plateaus, failing to scale like Large Language Models (LLMs).",
          "While generative recommendation (using Semantic IDs and Transformers) shows promise, advanced industrial implementations remain proprietary, preventing open validation of scaling laws on public datasets.",
          "The academic community lacks a transparent, efficient post-training pipeline for generative recommenders to achieve competitive performance."
        ],
        "solution": [
          "Introduces MiniOneRec, a fully open-source, end-to-end generative recommendation framework built on a Qwen-based LLM backbone.",
          "Employs Residual Quantized Variational Autoencoder (RQ-VAE) for item tokenization into compact, hierarchical 3-level Semantic IDs (SIDs).",
          "Aligns SIDs with the LLM through vocabulary augmentation and joint supervised fine-tuning (SFT) tasks combining recommendation and semantic alignment objectives.",
          "Optimizes preferences using Group Relative Policy Gradient (GRPO) with a hybrid reward function and constrained beam search to ensure diverse and valid recommendations."
        ],
        "keyInsights": [
          "Generative recommendation, when structured with Semantic IDs (SIDs), demonstrably exhibits LLM-like scaling laws on public datasets, with larger models consistently achieving lower losses.",
          "Full-process Semantic ID (SID) alignment throughout training is crucial for effectively integrating the LLM's world knowledge and achieving high performance in recommendation tasks.",
          "Reinforcement learning, specifically with a hybrid reward function and constrained beam search, can effectively refine generative policies and uncover generalizable interaction patterns, enabling transferability across domains."
        ],
        "results": [
          "Empirically validated scaling laws for generative recommendation, demonstrating consistent loss reduction and faster convergence on public Amazon datasets as Qwen backbone models scaled from 0.5B to 7B parameters.",
          "Consistently surpassed all traditional, generative, and LLM-based baselines across HR@K and NDCG@K metrics on both Industrial and Office Amazon Review datasets.",
          "Demonstrated superior transferability and robustness, with MiniOneRec achieving competitive accuracy on unseen domains by effectively discovering generalizable SID interaction patterns.",
          "Ablation studies confirmed the full-process SID alignment strategy, constrained beam search sampling, and the hybrid reward design as key contributors to its top performance."
        ]
      },
      "image_url": "image/2510.24431v1.png",
      "universal_paper_id": "2510.24431",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 153,
          "last_7_days": 153
        },
        "public_total_votes": 25
      },
      "first_publication_date": "2025-10-28T13:58:36.000Z",
      "publication_date": "2025-10-28T13:58:36.000Z",
      "updated_at": "2025-10-29T04:37:00.300Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.IR",
        "efficient-transformers",
        "fine-tuning",
        "generative-models",
        "parameter-efficient-training",
        "recommender-systems",
        "reinforcement-learning",
        "representation-learning",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 14,
      "github_url": "https://github.com/AkaliKong/MiniOneRec",
      "distance": 1
    }
  ],
  "page": 0
};