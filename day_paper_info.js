const papersData = {
  "papers": [
    {
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们介绍Kimi Linear，这是一种混合线性注意力架构，首次在公平比较的各种场景下超越了完整注意力——包括短上下文、长上下文和强化学习（RL）扩展范式。其核心是Kimi Delta Attention（KDA），这是一个富有表现力的线性注意力模块，它扩展了Gated DeltaNet，引入了更细致的门控机制，使得有限的有限状态RNN内存得以更有效地利用。我们的定制化分块算法通过一种特化的对角加低秩（DPLR）转移矩阵实现了高硬件效率，与普通的DPLR公式相比，显著减少了计算量，同时依然更加符合经典的增量规则。\n\n我们预训练了一个具有30亿激活参数和480亿总参数的Kimi Linear模型，基于KDA和多头潜在注意力（MLA）的层级混合。我们的实验表明，在相同的训练方案下，Kimi Linear在所有评估任务中都以可观的优势超越了完整的MLA，同时将KV缓存的使用减少了多达75%，在1M上下文的解码吞吐量上提高了最多6倍。这些结果表明，Kimi Linear可以作为完整注意力架构的直接替代，提供更卓越的性能和效率，包括更长输入和输出长度的任务。\n\n为了支持进一步研究，我们开源了KDA内核和vLLM实现，并发布了预训练和指令调优的模型检查点。",
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
        "total_votes": 10,
        "visits_count": {
          "all": 314,
          "last_7_days": 314
        },
        "public_total_votes": 39
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
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们推出了 Tongyi DeepResearch，这是一个具有自主研究能力的大型语言模型，专门设计用于长时间、深入的信息寻求研究任务。为了激励自主深入研究能力，Tongyi DeepResearch 通过一个端到端的训练框架进行开发，该框架结合了中期训练和后期训练的自主性，能够在复杂任务中实现可扩展的推理和信息寻求。我们设计了一个高度可扩展的数据合成管道，该管道是完全自动化的，不依赖于昂贵的人力标注，并赋能于所有训练阶段。通过为每个阶段构建定制的环境，我们的系统能够在整个过程中实现稳定和一致的交互。Tongyi DeepResearch 拥有总计 305 亿个参数，每个 token 仅激活 33 亿个参数，在一系列自主深度研究基准测试中取得了最先进的性能，包括 Humanity's Last Exam、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES 和 xbench-DeepSearch-2510。我们开源了模型、框架以及完整的解决方案，以赋能社区。",
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
        "total_votes": 44,
        "visits_count": {
          "all": 1602,
          "last_7_days": 1602
        },
        "public_total_votes": 126
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
      "abstract": "大语言模型（LLMs）的强化学习（RL）微调常因训练和推理策略之间的数值不匹配而导致不稳定。尽管之前的研究尝试通过算法修正或工程对齐来缓解此问题，但我们发现其根本原因在于浮点数精度本身。尽管广泛采用的BF16具有较大的动态范围，但它引入了较大的舍入误差，从而破坏了训练与推理之间的一致性。在本研究中，我们证明简单地恢复使用\\textbf{FP16}有效地消除了这种不匹配。这一改变简单，现代框架完全支持，仅需少量代码更改，而且无需修改模型架构或学习算法。我们的结果表明，使用FP16在不同任务、算法和框架中均能产生更稳定的优化、更快的收敛速度和更强的性能。我们希望这些发现能够激励对强化学习微调中的精度权衡进行更广泛的重新考虑。",
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
        "total_votes": 16,
        "visits_count": {
          "all": 357,
          "last_7_days": 357
        },
        "public_total_votes": 51
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
      "abstract": "现代的大型语言模型（LLM）主要通过显式文本生成进行“思考”，例如链式思维（CoT），这将推理推迟到训练后，并在一定程度上未充分利用预训练数据。我们提出并开源了Ouro，命名为递归的乌罗波洛斯（Ouroboros），这是一类预训练的循环语言模型（LoopLM），它通过(i) 潜在空间中的迭代计算，(ii) 对学习的深度分配进行熵正则化的目标，和(iii) 扩展到7.7万亿个令牌，而在预训练阶段内构建推理。Ouro 1.4B和2.6B模型表现出优越的性能，能够与多达12B的最新状态大型语言模型在广泛基准测试中匹配。通过控制实验，我们表明这种优势不是来自知识容量的增加，而是来自更出色的知识处理能力。我们还展示了LoopLM所产生的推理痕迹与最终输出的对齐程度优于显式的链式思维。我们希望我们的结果能够展示LoopLM作为推理时代一种新型扩展方向的潜力。我们的模型可以在此找到：这个http URL。",
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
        "total_votes": 19,
        "visits_count": {
          "all": 673,
          "last_7_days": 673
        },
        "public_total_votes": 61
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
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们介绍了Emu3.5，一个大规模多模态世界模型，能够原生地预测视觉和语言的下一个状态。Emu3.5经过端到端的预训练，采用统一的下一个标记预测目标，在一个包含超过10万亿个标记的视觉-语言交错数据语料库上进行训练，这些数据主要来源于互联网视频的顺序帧和转录文本。该模型自然地接受交替的视觉-语言输入，生成交替的视觉-语言输出。Emu3.5还通过大规模强化学习进一步后训练，以增强多模态推理和生成。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐标记解码转换为双向并行预测，使每张图片的推理速度提高约20倍，而不牺牲性能。Emu3.5表现出强大的原生多模态能力，包括长时间范围的视觉-语言生成、任意到图像（X2I）生成以及复杂的富文本图像生成。它还表现出可推广的世界建模能力，能够在不同情境和任务中实现时空一致的世界探索和开放世界的具身操作。相比之下，Emu3.5在图像生成和编辑任务上达到了与Gemini 2.5 Flash Image（Nano Banana）可比的性能，并在一系列交互生成任务上展现出更优的结果。我们在此链接开源Emu3.5，以支持社区研究。",
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
        "total_votes": 13,
        "visits_count": {
          "all": 228,
          "last_7_days": 228
        },
        "public_total_votes": 36
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
      "abstract": "大型语言模型（LLMs）在需要多步推理的问题上常常表现不佳。对于小规模的开源模型，带可验证奖励的强化学习（RLVR）在即使经过多次尝试仍然很少采样到正确解时会失败，而监督微调（SFT）往往通过僵化的逐标记模仿导致对长示例的过拟合。为了解决这个问题，我们提出了监督强化学习（SRL），一个将问题解决重新构建为生成一系列逻辑“动作”的框架。SRL训练模型在每次执行动作之前生成内部推理独白。它根据模型的动作与从SFT数据集中提取的专家动作之间的相似性，以逐步的方式提供更平滑的奖励。这种监督即使在所有回归都不正确的情况下也提供了更丰富的学习信号，同时鼓励以专家示范为指导的灵活推理。因此，SRL使小模型能够学习以前无法通过SFT或RLVR学习的挑战性问题。此外，在使用RLVR进行微调之前用SRL初始化训练可获得最佳的整体性能。除了推理基准外，SRL在代理软件工程任务上也能有效泛化，确立了其作为面向推理的LLMs的强大和多功能训练框架。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 226,
          "last_7_days": 226
        },
        "public_total_votes": 25
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
      "id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "paper_group_id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "title": "Deep sequence models tend to memorize geometrically; it is unclear why",
      "abstract": "在序列建模中，原子事实的参数化记忆主要被抽象为实体之间共现的粗暴查找。我们将这种关联视角与记忆存储的几何视角进行对比。我们首先孤立出一个清晰且可分析的Transformer推理实例，该实例与记忆严格作为训练期间指定的局部共现的存储方式不兼容。相反，模型必须以某种方式合成其自身的原子事实几何结构，编码所有实体之间的全局关系，包括那些未共现的实体。这反过来简化了一个涉及$\\ell$重组合的困难推理任务，使其成为一个易于学习的一步几何任务。\n\n从这一现象中，我们提取出难以解释的神经嵌入几何的基本方面。我们认为，这种几何的出现，尽管仅优化局部关联，但并不能简单归因于典型的架构或优化压力。相反地，即使这种几何并不比对关联的粗暴查找更简洁，仍然学到了优雅的几何。\n\n然后，通过分析与Node2Vec的联系，我们展示了这种几何源于一种谱偏差——与现有理论相对比，这种偏差确实自然而然地出现，尽管缺乏各种压力。这一分析还为从业者指明了在使Transformer记忆更加强几何化方面的显著提升空间。我们希望参数化记忆的几何视角能够鼓励研究者重新审视指导知识获取、容量、发现和遗忘等领域的默认直觉。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 115,
          "last_7_days": 115
        },
        "public_total_votes": 19
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
      "id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "paper_group_id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "title": "The Era of Agentic Organization: Learning to Organize with Language Models",
      "abstract": "我们设想一个新的人工智能时代，称为代理组织，在这个时代，智能体通过合作和并行工作解决复杂问题，从而实现超越个体智能的成果。为了实现这一愿景，我们引入异步思维（AsyncThink）作为与大型语言模型推理的新范式，它将内部思维过程组织成可并行执行的结构。具体而言，我们提出一种思维协议，组织者动态地将子查询分配给工作者，合并中间知识，并生成连贯的解决方案。更重要的是，这一协议中的思维结构可以通过强化学习进一步优化。实验表明，AsyncThink在推理延迟方面比并行思维降低了28%，同时提高了数学推理的准确性。此外，AsyncThink能够推广其学习的异步思维能力，有效应对未见过的任务而无需额外训练。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 147,
          "last_7_days": 147
        },
        "public_total_votes": 18
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
      "id": "019a390a-6398-7cfc-8161-edd680691708",
      "paper_group_id": "019a390a-6398-7cfc-8161-edd680691708",
      "title": "The End of Manual Decoding: Towards Truly End-to-End Language Models",
      "abstract": "“端到端”标签对于大型语言模型（LLM）来说是一个误称。实际上，它们依赖于一种不可微分的解码过程，需要繁琐的手动调节超参数，如温度和top-p值。本文介绍了AutoDeco，这是一种新颖的架构，通过学习控制自身的解码策略，实现真正的“端到端”生成。我们在标准变压器上增加了轻量级头部，在每一步中，动态预测上下文特定的温度和top-p值以及下一个标记的logits。这种方法将解码转变为一个参数化的标记级过程，使模型能够在单次前向传递中自我调节采样策略。\n\n通过在八个基准上的广泛实验，我们证明了AutoDeco不仅显著优于默认解码策略，而且在性能上与通过“破解测试集”得出的理想调优基线相当—这是任何静态方法的实际上限。重要的是，我们发现了一种基于指令的解码控制的新兴能力：模型学习理解自然语言指令（例如，“以低随机性生成”），并根据每个标记调整其预测的温度和top-p，为可调整和交互式LLM解码开辟了新范式。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 96,
          "last_7_days": 96
        },
        "public_total_votes": 17
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
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我提升系统需要与环境互动以实现持续适应。我们介绍了SPICE（自我对弈语料环境），这是一个强化学习框架，在其中单一模型扮演两个角色：挑战者，从一个大型语料库中提取文档以生成多样化的推理任务；推理者，解决这些任务。通过对抗动态，挑战者在推理者能力的最前沿创建了自动课程，而语料库的基础提供了丰富且几乎用不完的外部信号，这是持续改进所必需的。与现有的无基础自我对弈方法相比，后者所提供的好处更为有限，SPICE在多个模型家族的数学（+8.9%）和一般推理（+9.8%）基准测试中实现了一致的提升。我们的分析揭示了文档基础在SPICE中是一个关键因素，它不断生成越来越具挑战性的目标并实现这些目标，从而实现持续自我提升。",
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
          "all": 241,
          "last_7_days": 241
        },
        "public_total_votes": 36
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
      "id": "019a3953-42a0-7675-8f47-eef2729102dc",
      "paper_group_id": "019a3953-42a0-7675-8f47-eef2729102dc",
      "title": "Active Learning with Task-Driven Representations for Messy Pools",
      "abstract": "主动学习在处理混乱、未经整理的样本池时具有特别的潜力，因为数据点与目标任务的相关性各异。然而，当前针对该问题的最先进方法依赖于使用固定的无监督样本池表示，主要集中于修改获取函数。我们表明，这种模型设置可能会削弱其在处理混乱样本池时的有效性，因为此类表示可能无法捕捉与任务相关的重要信息。为了解决这个问题，我们建议使用任务驱动的表示，这些表示在主动学习过程中会定期更新，使用以前收集的标签。我们提出了两种特定的学习这些表示的策略，一种基于直接学习半监督表示，另一种基于对初始无监督表示进行监督微调。我们发现这两种方法在实证性能上显著优于使用无监督或预训练表示。",
      "paper_summary": null,
      "image_url": "image/2510.25926v1.png",
      "universal_paper_id": "2510.25926",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 25,
          "last_7_days": 25
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-10-29T19:54:04.000Z",
      "publication_date": "2025-10-29T19:54:04.000Z",
      "updated_at": "2025-10-31T08:12:29.216Z",
      "topics": [
        "Computer Science",
        "cs.LG"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于大型语言模型的网络代理在信息获取方面展现出巨大的潜力，但在长期任务中的有效性受到上下文管理的基本权衡的制约。当前基于ReAct的代理由于积累了嘈杂的原始历史而遭遇上下文饱和，而固定地在每一步总结全部历史的方法则有可能不可逆地丢失关键细节。为了解决这些问题，我们提出了AgentFold，这是一种新的代理范式，专注于主动的上下文管理，灵感来自于人类认知过程中的回顾性整合。AgentFold将其上下文视为一个动态的认知工作空间，需要主动塑造，而不是一个被动的日志。在每一步中，它学习执行一种“折叠”操作，在多个尺度上管理其历史轨迹：它可以进行细致的凝聚，以保留重要的细微细节，或进行深度整合，以抽象掉整个多步骤子任务。在显著的基准测试中，成绩引人瞩目：通过简单的监督微调（不需要持续的预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上取得了36.2%的成绩，在BrowseComp-ZH上取得了47.3%的成绩。值得注意的是，这一表现不仅超越或匹配了大规模开源模型（如DeepSeek-V3.1-671B-A37B），还超过了领先的专有代理如OpenAI的o4-mini。",
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
        "total_votes": 7,
        "visits_count": {
          "all": 319,
          "last_7_days": 319
        },
        "public_total_votes": 42
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
      "id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "paper_group_id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "title": "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender",
      "abstract": "在推荐系统中，扩大特征交互模块（例如Wukong、RankMixer）或用户行为序列模块（例如LONGER）已取得显著成功。然而，这些努力通常是分开的，这不仅阻碍了双向信息交换，还使得统一优化和扩展变得困难。在本文中，我们提出了OneTrans，一种统一的Transformer主干，能够同时进行用户行为序列建模和特征交互。OneTrans采用统一的分词器，将顺序和非顺序属性转换为单一的令牌序列。堆叠的OneTrans块在相似的顺序令牌之间共享参数，同时为非顺序令牌分配特定的参数。通过因果注意力和跨请求的KV缓存，OneTrans实现了中间表示的预计算和缓存，在训练和推理过程中显著降低了计算成本。工业规模数据集上的实验结果表明，OneTrans在参数增加时能够有效扩展，始终优于强基准，并在在线A/B测试中实现每用户GMV提升5.68%。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 102,
          "last_7_days": 102
        },
        "public_total_votes": 18
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
      "id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "paper_group_id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "title": "Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model",
      "abstract": "最近的大型语言模型（LLM）研究经历了从编码-解码建模到如今主流的仅解码建模的架构转变。然而，这一快速的过渡缺乏严格的比较分析，尤其是从规模的角度来看，这引发了人们对编码-解码模型潜力被忽视的担忧。为填补这一空白，我们重新审视了编码-解码 LLM（RedLLM），并利用最新的仅解码 LLM（DecLLM）的方法进行了增强。我们在不同模型规模（从约150M到约8B）上，对使用前缀语言建模（LM）进行预训练的RedLLM和使用因果LM进行预训练的DecLLM进行了全面比较。我们的实验使用RedPajama V1（1.6T 令牌）进行预训练，并使用FLAN进行指令调整，结果显示RedLLM展现出引人注目的扩展特性和令人惊讶的强大性能。尽管DecLLM在预训练期间总体上更具计算优化性，但RedLLM在扩展性和上下文长度外推能力上表现出可比性。经过指令调整后，RedLLM在各种下游任务上取得了可比甚至更好的结果，同时享有显著更好的推理效率。我们希望我们的发现能够激励更多的人重新审视RedLLM，释放其在开发强大高效的LLM方面的潜力。",
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
          "all": 35,
          "last_7_days": 35
        },
        "public_total_votes": 8
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
      "id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "paper_group_id": "019a383a-6720-7f94-9278-16f8e0ab57e9",
      "title": "Running VLAs at Real-time Speed",
      "abstract": "在本文中，我们展示了如何使用单个消费级GPU以30Hz的帧率和最多480Hz的轨迹频率运行pi0级别的多视角VLA。这使得以前被认为大型VLA模型无法实现的动态和实时任务成为可能。为了实现这一目标，我们引入了一系列策略来消除模型推理中的开销。真实世界的实验表明，采用我们策略的pi0策略在抓取掉落的笔的任务中取得了100%的成功率。基于这些结果，我们进一步提出了一个用于实时机器人控制的全流式推理框架。代码可在此HTTPS网址获取。",
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
          "all": 58,
          "last_7_days": 58
        },
        "public_total_votes": 11
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
      "id": "019a33e4-dd46-72ef-8bca-cc5b5c1591d2",
      "paper_group_id": "019a33e4-dd46-72ef-8bca-cc5b5c1591d2",
      "title": "VFXMaster: Unlocking Dynamic Visual Effect Generation via In-Context Learning",
      "abstract": "视觉特效（VFX）对于数字媒体的表现力至关重要，但其制作仍然是生成性人工智能面临的主要挑战。现行方法通常依赖于“一种效应一种LoRA”的范式，这种方法资源消耗大，并且在本质上无法推广到未见过的特效，从而限制了可扩展性和创作。为了解决这一挑战，我们推出了VFXMaster，这是第一个统一的基于参考的VFX视频生成框架。它将特效生成重新定义为一种上下文学习任务，使其能够从参考视频中向目标内容复制多种动态特效。此外，它在未见过的特效类别上表现出显著的泛化能力。具体来说，我们设计了一种上下文条件策略，通过参考示例提示模型。同时，设计了一种上下文注意力掩码，以精确解耦和注入基本的特效属性，使单一的统一模型能够在不出现信息泄漏的情况下掌握特效模仿。此外，我们还提出了一种高效的一次性效果适配机制，能够快速提高对来自单个用户提供的视频中的困难未见效果的泛化能力。大量实验表明，我们的方法有效地模仿了多种类别的特效信息，并在跨领域效果上表现出卓越的泛化能力。为了促进未来的研究，我们将向社区发布我们的代码、模型和一个综合数据集。",
      "paper_summary": {
        "summary": "VFXMaster introduces a unified, reference-based framework for dynamic visual effect generation that leverages in-context learning to achieve strong generalization to unseen effects. The model outperforms prior methods in effect fidelity and dynamic realism while significantly reducing content leakage, enabling the creation of diverse visual effects from a single target image and a reference video.",
        "originalProblem": [
          "Existing video generation models struggle with dynamic visual effects (VFX) due to data scarcity, complex, unstructured dynamics, and limited controllability with traditional spatial conditions.",
          "Prior AI-driven VFX generation methods, like \"one-LoRA-per-effect\" paradigms, face severe scalability issues, requiring dedicated training for each effect and limiting generalization to out-of-domain categories.",
          "The challenge of disentangling essential effect dynamics from irrelevant content (e.g., subjects, backgrounds) in reference videos often leads to undesirable information leakage during transfer."
        ],
        "solution": [
          "VFXMaster redefines VFX generation as an in-context learning task, using a unified model that learns to imitate general effect dynamics from a reference video rather than memorizing individual effects.",
          "A specialized in-context attention mask is designed to precisely control information flow within the Diffusion Transformer, ensuring selective transfer of effect attributes while preventing unwanted content leakage from the reference video.",
          "An efficient one-shot effect adaptation mechanism is introduced, utilizing learnable concept-enhancing tokens (`z_ce`) fine-tuned on a single user-provided example to rapidly boost performance on challenging out-of-domain (OOD) effects."
        ],
        "keyInsights": [
          "Reframing VFX generation as an in-context learning problem allows a single model to learn a general capability for effect imitation, significantly improving scalability and generalization compared to per-effect training.",
          "Careful control of attention flow via a dedicated mask is crucial for disentangling visual effect attributes from other video content, preventing information leakage and ensuring accurate effect transfer.",
          "Despite strong in-context generalization, a lightweight, one-shot adaptation mechanism can effectively capture fine-grained characteristics of novel out-of-domain effects with minimal data and computational cost."
        ],
        "results": [
          "VFXMaster consistently achieved the lowest Fréchet Video Distance (FVD) of 1369 and the highest Dynamic Degree of 0.80 on in-domain effects, outperforming all baseline methods including OmniEffects.",
          "The model demonstrated strong generalization to out-of-domain effects, with the one-shot adaptation mechanism substantially improving Effect Fidelity Score (EFS) from 0.47 to 0.70 and Content Leakage Score (CLS) from 0.79 to 0.87.",
          "Ablation studies confirmed the critical role of the in-context attention mask, with its removal causing drastic performance degradation (EFS dropped to 0.11, CLS to 0.24), highlighting its effectiveness in disentangling effect information."
        ]
      },
      "image_url": "image/2510.25772v1.png",
      "universal_paper_id": "2510.25772",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 41,
          "last_7_days": 41
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-29T17:59:53.000Z",
      "publication_date": "2025-10-29T17:59:53.000Z",
      "updated_at": "2025-10-30T06:53:48.230Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.CV",
        "few-shot-learning",
        "generative-models",
        "representation-learning",
        "transfer-learning",
        "transformers",
        "video-understanding"
      ],
      "organization_info": [
        {
          "name": "Dalian University of Technology",
          "image": null
        },
        {
          "name": "Kuaishou Technology",
          "image": null
        },
        {
          "name": "Oxford University",
          "image": null
        },
        {
          "name": "ZMO AI Inc.",
          "image": null
        },
        {
          "name": "Kling Team, Kuaishou Technology",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 9,
      "github_url": "https://github.com/libaolu312/VFXMaster",
      "distance": 1
    },
    {
      "id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "paper_group_id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "title": "Context Engineering 2.0: The Context of Context Engineering",
      "abstract": "卡尔·马克思曾写道“人类的本质是社会关系的总和”，这表明个体并非孤立的实体，而是从根本上受到与其他实体之间互动的影响，其中背景扮演着构成性和本质性的角色。随着计算机和人工智能的出现，这些背景不再仅限于纯粹的人际互动：人机互动也被纳入其中。由此，一个核心问题浮现：机器如何才能更好地理解我们的情境和目的？为了解决这一挑战，研究人员最近提出了“背景工程”的概念。尽管它通常被视为代理时代的一项新创新，但我们认为相关的实践可以追溯到二十多年前。自1990年代初以来，该领域经历了不同的历史阶段，每一个阶段都受到机器智能水平的影响：从围绕原始计算机建立的早期人机交互框架，到以智能代理驱动的当今人－代理交互范式，以及未来可能出现的人类水平或超人类智能。在本文中，我们定位背景工程，提供系统定义，概述其历史和概念背景，并考察实践中的关键设计考虑。通过解决这些问题，我们希望为背景工程提供一个概念基础，并描绘其未来的光明前景。本文是朝着系统化AI系统背景工程的更广泛社区努力的一个起点。",
      "paper_summary": null,
      "image_url": "image/2510.26493v1.png",
      "universal_paper_id": "2510.26493",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 15,
          "last_7_days": 15
        },
        "public_total_votes": 2
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
      "id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "paper_group_id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "title": "Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark",
      "abstract": "最近的视频生成模型能够生成高保真的、时间一致性强的视频，这表明它们可能编码了大量的世界知识。除了现实的合成外，它们还表现出一些新兴行为，表明具备视觉感知、建模和处理能力。然而，一个重要的问题仍然存在：视频模型是否准备好在具有挑战性的视觉推理场景中作为零-shot 推理者？在这项工作中，我们进行了一项实证研究，全面调查这一问题，重点关注领先且受欢迎的 Veo-3。我们在12个维度上评估其推理行为，包括空间、几何、物理、时间和具身逻辑，系统地描绘其优点和失败模式。为了标准化这项研究，我们将评估数据整理成 MME-CoF，一个紧凑的基准，能够深入和全面地评估帧链（CoF）推理。我们的发现表明，尽管当前的视频模型在短期空间一致性、精细的基底和局部一致性动态方面表现出有希望的推理模式，但在长期因果推理、严格的几何约束和抽象逻辑方面仍然有限。总体而言，它们尚不可靠作为独立的零-shot 推理者，但作为专门推理模型的补充视觉引擎展现出令人鼓舞的迹象。项目页面：这个 https URL",
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
          "all": 51,
          "last_7_days": 51
        },
        "public_total_votes": 12
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
      "id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "paper_group_id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "title": "$π_\\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models",
      "abstract": "视觉-语言-动作（VLA）模型使机器人能够理解并执行来自多模态输入的复杂任务。尽管最近的研究探讨了使用强化学习（RL）来自动化数据收集过程，以扩展监督微调（SFT），但由于来自迭代去噪的不易处理的动作对数似然，应用大规模RL于基于流的VLA（例如，$\\pi_0$，$\\pi_{0.5}$）仍然具有挑战性。我们通过$\\pi_{\\text{RL}}$来应对这一挑战，它是一个用于在并行仿真中训练基于流的VLA的开源框架。$\\pi_{\\text{RL}}$实现了两种RL算法：（1）{Flow-Noise}将去噪过程建模为一个具有可学习噪声网络的离散时间马尔可夫决策过程（MDP），以便精确计算对数似然。（2）{Flow-SDE}将去噪与代理-环境交互结合，形成一个二层MDP，利用常微分方程（ODE）到随机微分方程（SDE）的转换进行高效的RL探索。我们在LIBERO和ManiSkill基准上评估了$\\pi_{\\text{RL}}$。在LIBERO上，$\\pi_{\\text{RL}}$将少样本SFT模型$\\pi_0$和$\\pi_{0.5}$的表现分别从57.6%提升到97.6%和从77.1%提升到98.3%。在ManiSkill中，我们在320个并行环境中训练$\\pi_{\\text{RL}}$，使得$\\pi_0$在4352个拣取和放置任务中，从41.6%提高到85.7%，$\\pi_{0.5}$从40.0%提高到84.8%，展现了在异构仿真下可扩展的多任务RL。总体而言，$\\pi_{\\text{RL}}$在性能和泛化能力上超越了SFT模型，验证了在线RL在基于流的VLA中的有效性。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 31,
          "last_7_days": 31
        },
        "public_total_votes": 6
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
      "id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "paper_group_id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "title": "Uniform Discrete Diffusion with Metric Path for Video Generation",
      "abstract": "连续空间视频生成技术迅速发展，而离散方法由于误差积累和长时间上下文不一致性而滞后。本文回顾了离散生成建模，并提出了Uniform discRete diffuSion with metric pAth（URSA），这是一个简单但强大的框架，它填补了与连续方法之间的差距，实现可扩展的视频生成。URSA的核心将视频生成任务定义为对离散时空标记的迭代全局优化。它融合了两个关键设计：线性化度量路径和分辨率依赖的时间步移机制。这些设计使得URSA能够高效扩展到高分辨率图像合成和长时间视频生成，同时所需的推理步骤显著减少。此外，我们还引入了一种异步时间微调策略，将多种任务统一在单一模型中，包括插值和图像到视频的生成。在具有挑战性的视频和图像生成基准上的大量实验表明，URSA consistently超越现有的离散方法，且性能与最先进的连续扩散方法相媲美。代码和模型可在此HTTPS URL获取。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 286,
          "last_7_days": 286
        },
        "public_total_votes": 35
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
      "id": "019a362d-6c5a-7a32-96c6-e60ae4941758",
      "paper_group_id": "019a362d-6c5a-7a32-96c6-e60ae4941758",
      "title": "What Really Matters in Matrix-Whitening Optimizers?",
      "abstract": "最近出现了一系列优化器，它们以不同的方式近似相同的“矩阵去相关”变换。在本研究中，我们系统性地解构了这些优化器，旨在揭示解释性能的关键组成部分。在各个调优超参数下，所有类型的矩阵去相关方法都可靠地超越了逐元素的对应物，例如Adam。矩阵去相关通常与谱下降相关——然而，实验表明，性能的提升*并非仅由准确的谱归一化所解释*——特别是，SOAP在每步的增益最大，即使Muon在最陡的谱下降方向上下降得更准确。我们认为，矩阵去相关有两个目的，而矩阵去相关中的方差自适应成分是解释这一性能差距的被忽视的要素。实验表明，经过方差自适应处理的优化器版本始终优于它们的符号下降对应物，包括Muon的自适应版本。我们进一步分析了方差自适应策略，发现尽管前瞻式近似并不那么有效，但低秩方差估计器能够有效降低内存成本而不损失性能。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 71,
          "last_7_days": 71
        },
        "public_total_votes": 12
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
      "abstract": "强化学习（RL）在提高大型语言模型（LLM）的推理能力方面展现出了显著的潜力。然而，RL在LLM中的成功在很大程度上依赖于人工策划的数据集和可验证的奖励，这限制了它们的可扩展性和一般性。最近的自我对弈RL方法受到游戏和围棋成功的启发，旨在无需人工标注数据来增强LLM的推理能力。然而，它们的方法主要依赖于一个基于反馈的环境（例如，Python解释器或游戏引擎）；将其扩展到一般领域仍然具有挑战性。为了应对这些挑战，我们提出了多智能体演化（MAE）框架，使得LLM能够在解决各种任务（包括数学、推理和一般知识问答）中自我演化。MAE的核心设计基于一组三个交互代理（提议者、求解者、评估者），它们由单个LLM实例化，并应用强化学习来优化其行为。提议者生成问题，求解者尝试解决方案，评估者在共同演化的过程中对这两者进行评估。针对Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试中实现了平均4.54%的提升。这些结果突显了MAE作为一种可扩展、数据有效的方法，在最小依赖人工策划监督的情况下提高LLM的一般推理能力。",
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
        "total_votes": 16,
        "visits_count": {
          "all": 690,
          "last_7_days": 690
        },
        "public_total_votes": 69
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
      "id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "paper_group_id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "title": "Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks",
      "abstract": "人类拥有空间推理能力，使他们能够通过多模态观察（如视觉和声音）理解空间。大型多模态推理模型通过学习感知和推理扩展了这些能力，在各种空间任务中表现出色。然而，对于这些模型的系统评估和公开基准仍然有限。在本次调查中，我们对大型模型的多模态空间推理任务进行了全面回顾，分类了多模态大型语言模型（MLLMs）的近期进展，并介绍了用于评估的开放基准。我们首先概述了一般空间推理，重点讨论了后训练技术、可解释性和架构。除了经典的2D任务之外，我们还考察了空间关系推理、场景和布局理解，以及视觉问答和在3D空间中的定位。我们还回顾了体态人工智能的进展，包括视觉语言导航和动作模型。此外，我们考虑了音频和自我中心视频等新兴模态，这些模态通过新传感器为新型空间理解做出了贡献。我们相信，这项调查奠定了坚实的基础，并为不断增长的多模态空间推理领域提供了见解。有关本次调查的更新信息、代码和开放基准的实现可以在此 https URL 找到。",
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
          "all": 88,
          "last_7_days": 88
        },
        "public_total_votes": 18
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
      "id": "019a3ec1-89b1-7fe5-9a6c-ec85e5b19146",
      "paper_group_id": "019a3ec1-89b1-7fe5-9a6c-ec85e5b19146",
      "title": "CYPRESS: Crop Yield Prediction via Regression on Prithvi's Encoder for Satellite Sensing",
      "abstract": "准确及时的作物产量预测对全球粮食安全和现代农业管理至关重要。传统方法通常缺乏精准农业所需的可扩展性和颗粒度。本文介绍了CYPRESS（通过普里斯维编码器进行卫星感知的作物产量预测），这是一种旨在实现高分辨率、田间油菜产量预测的深度学习模型。CYPRESS利用一个预训练的大规模地理空间基础模型（Prithvi-EO-2.0-600M），并将其调整为连续回归任务，将多时相卫星图像转化为密集的像素级产量图。经过对来自加拿大大草原的综合数据集的评估，CYPRESS显示出优于现有基于深度学习的产量预测模型的卓越性能，突显了对基础模型进行微调在专业农业应用中的有效性。通过提供连续的高分辨率输出，CYPRESS为精准农业提供了一种比传统分类或县级聚合方法更具可操作性的工具。本研究验证了一种新颖的方法，弥合了大规模地球观测与农场决策之间的差距，为详细的农业监测提供了可扩展的解决方案。",
      "paper_summary": null,
      "image_url": "image/2510.26609v1.png",
      "universal_paper_id": "2510.26609",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 14,
          "last_7_days": 14
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-30T15:37:40.000Z",
      "publication_date": "2025-10-30T15:37:40.000Z",
      "updated_at": "2025-11-01T09:31:02.449Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.LG",
        "eess.IV",
        "Electrical Engineering and Systems Science",
        "multi-modal-learning",
        "representation-learning",
        "semantic-segmentation",
        "time-series-analysis",
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
      "id": "019a3fd5-f17a-78da-b7e2-a84569676df2",
      "paper_group_id": "019a3fd5-f17a-78da-b7e2-a84569676df2",
      "title": "Accelerating mathematical research with language models: A case study of an interaction with GPT-5-Pro on a convex analysis problem",
      "abstract": "最近在大型语言模型方面的进展使它们在数学研究中变得越来越有能力。然而，随着它们推理能力的提高，评估它们的数学能力变得愈加具有挑战性。用于评估的问题必须既不能太简单，也不能太困难，它们的表现已经无法通过一个单一的数字分数来总结，且有意义的评估需要专家的监督。\n在这项工作中，我们研究了作者与大型语言模型之间的互动，证明一个与凸优化相关的引理。具体来说，我们在严格凸函数周围建立了双共轭算子的梯度的泰勒展开式——即通过应用芬蒵变换两次得到的算子，在OpenAI最新模型GPT-5-pro的协助下完成。\n除了我们并不确定其新颖性的数学结果之外，我们的主要贡献在于记录这种协作推理过程。GPT-5-pro通过建议相关的研究方向和证明一些中间结果加速了我们的进展。然而，它的推理仍然需要细致的监督，特别是在纠正细微错误方面。尽管限于一个单独的数学问题和一个单一的语言模型，这个实验体现了大型语言模型作为数学合作伙伴的潜力和当前的局限性。",
      "paper_summary": null,
      "image_url": "image/2510.26647v1.png",
      "universal_paper_id": "2510.26647",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 12,
          "last_7_days": 12
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-30T16:15:01.000Z",
      "publication_date": "2025-10-30T16:15:01.000Z",
      "updated_at": "2025-11-01T14:32:56.954Z",
      "topics": [
        "Mathematics",
        "math.OC"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3376-efec-7ec9-a308-b79d26152ee9",
      "paper_group_id": "019a3376-efec-7ec9-a308-b79d26152ee9",
      "title": "Task Completion Agents are Not Ideal Collaborators",
      "abstract": "目前对智能体的评估仍然集中在单次任务完成上，未能考虑许多现实问题固有的迭代和协作特性，其中人类目标往往不明确且会发展变化。我们主张应从构建和评估任务完成智能体转向开发协作智能体，不仅通过其最终输出的质量来评估，还要评估其在整个问题解决过程中与人类努力的互动及提升效果。为了支持这一转变，我们引入了协作努力扩展框架，捕捉智能体的效用如何随着用户参与的增加而增长。通过案例研究和模拟评估，我们展示了最先进的智能体在多轮现实场景中的表现通常不佳，揭示了智能体设计中缺失的一个因素：维持参与度和支撑用户理解的能力。协作努力扩展为诊断智能体行为和指导开发更有效互动提供了一个视角。",
      "paper_summary": {
        "summary": "A new framework, \"collaborative effort scaling,\" is introduced to evaluate large language model agents based on their ability to leverage and enhance human effort in iterative problem-solving. It demonstrates that current LLM agents often fail to sustain productive collaboration, with some advanced models not benefiting from interaction, and optimal collaboration strategies varying by model capabilities.",
        "originalProblem": [
          "LLM agents are predominantly evaluated on a \"task completion\" paradigm, focusing on single-shot or autonomous final outputs.",
          "This paradigm fails to address real-world tasks that are iterative, collaborative, and feature underspecified or evolving human goals.",
          "Current agents frequently produce sub-optimal outcomes, struggle with feedback, lack transparency, and cause user frustration in multi-turn interactions."
        ],
        "solution": [
          "Proposed the \"collaborative effort scaling\" framework to evaluate agents on their ability to leverage and enhance human effort across interaction trajectories.",
          "Defined key properties for collaborative agents: interaction sustainability (increasing value with user effort) and maximum usability (sustaining engagement).",
          "Developed metrics (Overall Utility, Refinement Gain, Usability Drop) and applied them in a simulated travel planning environment using LLM-based user agents."
        ],
        "keyInsights": [
          "Optimizing AI agents solely for task completion leads to suboptimal performance in iterative, collaborative, and real-world scenarios.",
          "Effective collaboration strategies are model-dependent; structured approaches benefit less capable models but can hinder more advanced ones.",
          "Evaluating agents through \"collaborative effort scaling\" helps diagnose specific interaction failures, such as inefficient loops or poor feedback integration."
        ],
        "results": [
          "Current state-of-the-art agents (e.g., GPT-4o, Llama-3.1 70B) often fail to improve performance through collaboration, exhibiting inefficient interaction loops.",
          "Less capable models, such as Claude 3.5 Sonnet, showed a substantial performance boost with structured, two-stage collaboration strategies.",
          "More capable models like Claude 4.0 Sonnet did not benefit from a two-stage collaboration strategy, which led to a higher usability drop without proportional utility gains."
        ]
      },
      "image_url": "image/2510.25744v1.png",
      "universal_paper_id": "2510.25744",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 30,
          "last_7_days": 30
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-29T17:47:18.000Z",
      "publication_date": "2025-10-29T17:47:18.000Z",
      "updated_at": "2025-10-30T04:53:44.044Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "conversational-ai",
        "cs.AI",
        "cs.CL",
        "human-ai-interaction",
        "online-learning",
        "reasoning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/clinicalml/collaborative-effort-scaling",
      "distance": 1
    },
    {
      "id": "019a3fd6-036d-77d7-90bf-7bfba530f767",
      "paper_group_id": "019a3fd6-036d-77d7-90bf-7bfba530f767",
      "title": "AMO-Bench: Large Language Models Still Struggle in High School Math Competitions",
      "abstract": "我们推出了AMO-Bench，一个高级数学推理基准，难度达到数学奥林匹克级别或更高，由50个人工设计的问题组成。现有的基准广泛利用高中数学竞赛来评估大型语言模型（LLMs）的数学推理能力。然而，由于表现饱和，许多现有的数学竞赛对于评估顶级LLMs的效果正逐渐减弱（例如，AIME24/25）。为了解决这个问题，AMO-Bench通过确保所有50个问题都是（1）经过专家交叉验证以达到至少国际数学奥林匹克（IMO）难度标准，以及（2）完全原创的问题，以防止因数据记忆而导致的潜在性能泄漏，引入了更严格的挑战。此外，AMO-Bench中的每个问题只要求提供最终答案，而不是证明，从而实现自动和稳健的评分以进行评估。在对26个LLMs进行的实验结果中，即使是表现最好的模型在AMO-Bench上的准确率也仅为52.4%，大多数LLMs得分低于40%。除了这些较差的表现外，我们进一步的分析显示了随着测试时间计算能力增加，AMO-Bench上出现了一个有希望的扩展趋势。这些结果突显了当前LLMs在数学推理方面的显著改进空间。我们发布AMO-Bench以促进对提升语言模型推理能力的进一步研究。",
      "paper_summary": null,
      "image_url": "image/2510.26768v1.png",
      "universal_paper_id": "2510.26768",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 10,
          "last_7_days": 10
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-30T17:52:02.000Z",
      "publication_date": "2025-10-30T17:52:02.000Z",
      "updated_at": "2025-11-01T14:33:01.549Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 16,
      "github_url": "https://github.com/meituan-longcat/AMO-Bench",
      "distance": 1
    },
    {
      "id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "paper_group_id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "title": "GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning",
      "abstract": "受大型语言模型（LLMs）驱动的自主代理在复杂任务解决中的工具操作方面展现出了令人印象深刻的能力。然而，现有的范式如ReAct依赖于顺序推理和执行，未能利用独立子任务之间固有的并行性。这个顺序瓶颈导致了工具利用效率低下和多步骤推理场景中的表现亚optimal。我们引入了基于图的代理规划（GAP），一个新颖的框架，通过基于图的规划显式建模任务间依赖关系，以实现自适应的并行和串行工具执行。我们的方法训练代理基础模型，将复杂任务分解为以依赖关系为中心的子任务图，自动确定哪些工具可以并行执行，哪些必须遵循顺序依赖性。这种以依赖为中心的调度在执行效率和任务准确性上都取得了显著改善。为训练GAP，我们构建了一个高质量的数据集，其规划轨迹来自多跳问答（MHQA）基准。我们采用了两阶段的训练策略：在精心挑选的数据集上进行监督微调（SFT），然后在战略性抽样的查询上进行基于正确性的奖励函数强化学习（RL），在这些查询中，基于工具的推理提供了最大的价值。MHQA数据集上的实验结果表明，GAP显著超越了传统的ReAct基线，尤其是在多步骤检索任务上，同时通过智能并行化实现了工具调用效率的显著提高。项目页面可在此链接获取。",
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
          "all": 61,
          "last_7_days": 61
        },
        "public_total_votes": 10
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
      "id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "paper_group_id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "title": "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents",
      "abstract": "关于大规模监督微调人工智能代理的公共研究成果仍然相对稀少，因为收集代理训练数据面临独特的挑战。在本研究中，我们认为瓶颈并不是缺乏基础数据源，而是大量不同类型的数据分散在异构的格式、工具和接口中。为此，我们引入了代理数据协议（ADP），这是一种轻量级的表示语言，充当不同格式的代理数据集与下游统一代理训练流程之间的“中介语言”。ADP的设计足够表达，能够捕捉多种任务，包括API/工具使用、浏览、编码、软件工程和一般代理工作流程，同时保持易于解析和训练，而无需在每个数据集层面进行工程处理。在实验中，我们将13个现有的代理训练数据集统一转换为ADP格式，并将标准化的ADP数据转换为多个代理框架所需的训练格式。我们在这些数据上进行了监督微调（SFT），并展示了相较于相应基模型平均约20%的性能提升，在标准的编码、浏览、工具使用和研究基准上实现了最先进或接近最先进的性能，而无需特定领域的调优。所有代码和数据均已公开发布，希望ADP能够帮助降低标准化、可扩展和可重复的代理训练的门槛。",
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
        "total_votes": 7,
        "visits_count": {
          "all": 280,
          "last_7_days": 280
        },
        "public_total_votes": 35
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
      "id": "019a3a44-561c-793b-addd-93a35fdb5a05",
      "paper_group_id": "019a3a44-561c-793b-addd-93a35fdb5a05",
      "title": "From Embedding to Control: Representations for Stochastic Multi-Object Systems",
      "abstract": "本文研究如何在具有多个相互作用对象的随机非线性动力学中实现准确建模和有效控制。然而，不均匀的相互作用和随机拓扑使这一任务具有挑战性。我们通过提出\\textit{图可控嵌入}（GCE）来应对这些挑战，这是一个用于学习随机多对象动力学以进行线性控制的一般框架。具体而言，GCE基于希尔伯特空间嵌入，允许将受控随机动力学的概率分布直接嵌入到再生核希尔伯特空间（RKHS）中，这使得在其RKHS中进行线性操作的同时保留非线性表达能力。我们提供了关于GCE存在性、收敛性和适用性的理论保证。值得注意的是，采用了一种均值场近似技术，以高效捕捉对象之间的依赖关系并实现可证明的低样本复杂性。通过整合图神经网络，我们构建了能够适应动态交互模式并且能够在仅有有限训练实例的情况下推广到未见拓扑的数据依赖核特征。GCE能够无缝扩展到不同大小和拓扑的多对象系统。利用希尔伯特空间的线性特性，GCE还支持简单而有效的控制算法，以合成最优序列。在物理系统、机器人和电网上的实验验证了GCE，并在分布内和少样本测试中相较于各种竞争嵌入方法展示了一致的性能提升。",
      "paper_summary": {
        "summary": "Researchers from University College London developed Graph Controllable Embeddings (GCE) to model and control stochastic nonlinear dynamics in multi-object systems by learning representations that simplify dynamics into a form amenable to linear control. This framework achieved superior robustness and generalization, particularly in noisy, large-scale power grid environments, by explicitly addressing non-uniform interactions and stochasticity.",
        "originalProblem": [
          "Controlling complex, stochastic, and nonlinear multi-object systems with unknown models, where traditional model-based control is intractable.",
          "Existing controllable embedding methods struggle with stochastic dynamics, fail to account for relational topologies, and incur quadratic computational costs with object count.",
          "Graph Neural Networks (GNNs) excel at interaction modeling but typically learn embeddings that are not directly controllable in a linear or locally linear sense, complicating control synthesis."
        ],
        "solution": [
          "Proposes Graph Controllable Embeddings (GCE), integrating Hilbert space embeddings of conditional distributions (HSECD) with graph neural networks to learn an approximately linear latent space.",
          "Employs a disentangled feature representation and an adaptive mean-field approximation using Boltzmann-Gibbs weights to model non-uniform interactions and reduce computational complexity from O(N^2) to O(N).",
          "Applies a standard Linear Quadratic Regulator (LQR) in the learned, linearizable RKHS feature space for efficient and analytically tractable control synthesis."
        ],
        "keyInsights": [
          "Embedding conditional distributions into a Reproducing Kernel Hilbert Space (RKHS) effectively linearizes stochastic nonlinear dynamics, enabling the use of efficient linear control algorithms.",
          "An adaptive mean-field approximation with Boltzmann-Gibbs weights allows for efficient and accurate modeling of non-uniform inter-object interactions, crucial for generalization in dynamic topologies.",
          "The framework provides theoretical guarantees for consistency, convergence, and provably low sample complexity, validating its data efficiency and robustness for learning stochastic multi-object dynamics."
        ],
        "results": [
          "GCE consistently outperformed general controllable embedding methods (VAE, PCC) and graph-based approaches (GraphODE) across diverse control tasks, validating the need for control-specific multi-object embedding.",
          "The framework demonstrated superior robustness and generalization in few-shot and noisy conditions, notably achieving stable performance on challenging, large-scale power-grid environments where baselines failed.",
          "Experiments confirmed the benefits of adaptive non-uniform weighting and low sample complexity, showing that a Gaussian kernel for mean-field approximation provided the best performance and that GCE achieved the fastest evaluation time for predictions."
        ]
      },
      "image_url": "image/2510.26344v1.png",
      "universal_paper_id": "2510.26344",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 26,
          "last_7_days": 26
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-30T10:51:27.000Z",
      "publication_date": "2025-10-30T10:51:27.000Z",
      "updated_at": "2025-10-31T12:35:48.380Z",
      "topics": [
        "eess.SY",
        "Electrical Engineering and Systems Science"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    }
  ],
  "page": 0
};