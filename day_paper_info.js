const papersData = {
  "papers": [
    {
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们介绍了Kimi Linear，这是一种混合线性注意力架构，首次在公平比较的各种场景下超越了全注意力，包括短上下文、长上下文和强化学习（RL）扩展范围。其核心是Kimi Delta Attention（KDA），这是一个表达性强的线性注意力模块，通过更精细的门控机制扩展了Gated DeltaNet，使有限状态RNN内存的使用更加有效。我们量身定制的分块算法通过特定变体的对角加低秩（DPLR）转移矩阵实现了高硬件效率，与一般DPLR公式相比，显著减少了计算，同时与经典的delta规则保持更一致。\n\n我们预训练了一个Kimi Linear模型，具有30亿个激活参数和480亿个总参数，基于KDA和多头潜在注意力（MLA）按层混合。实验证明，在相同的训练方案下，Kimi Linear在所有评估任务中大幅超越了全MLA，同时将KV缓存使用量减少了多达75%，在1M上下文中实现了高达6倍的解码吞吐量。这些结果表明，Kimi Linear可以作为全注意力架构的替代品，具有更优的性能和效率，包括处理较长输入和输出长度的任务。\n\n为了支持进一步的研究，我们开源了KDA内核和vLLM实现，并发布了预训练和指令调优的模型检查点。",
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
        "total_votes": 31,
        "visits_count": {
          "all": 1057,
          "last_7_days": 1057
        },
        "public_total_votes": 82
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
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们介绍Emu3.5，一个大型多模态世界模型，可以原生地预测视觉和语言之间的下一个状态。Emu3.5采用端到端的方式，通过统一的下一个标记预测目标在包含超过10万亿个标记的视觉-语言交织数据语料库上进行预训练，这些数据主要来源于互联网视频的连续帧和转录文本。该模型自然接受交织的视觉-语言输入，并生成交织的视觉-语言输出。Emu3.5还通过大规模的强化学习进行后训练，以增强多模态推理和生成能力。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐标记解码转换为双向并行预测，提升每张图像的推理速度约20倍，而不牺牲性能。Emu3.5展现出强大的原生多模态能力，包括长时间跨度的视觉-语言生成、任意到图像（X2I）生成以及复杂的富文本图像生成。它还表现出可推广的世界建模能力，使其能够在多种场景和任务中进行时空一致的世界探索和开放世界的具身操控。作为对比，Emu3.5在图像生成和编辑任务上的表现可与Gemini 2.5 Flash Image（Nano Banana）相媲美，并在一系列交织生成任务中展示出更优的结果。我们将在这个https网址上开源Emu3.5，以支持社区研究。",
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
        "total_votes": 18,
        "visits_count": {
          "all": 450,
          "last_7_days": 450
        },
        "public_total_votes": 55
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
      "id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "paper_group_id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "title": "Defeating the Training-Inference Mismatch via FP16",
      "abstract": "强化学习（RL）对大型语言模型（LLM）的微调常常由于训练和推理策略之间的数值不匹配而遭遇不稳定性。尽管之前的研究尝试通过算法修正或工程对齐来缓解这一问题，但我们证明其根本原因在于浮点精度本身。尽管广泛采用的BF16具备较大的动态范围，但它引入的大量舍入误差破坏了训练与推理之间的一致性。在本研究中，我们展示了简单地恢复使用\\textbf{FP16}可以有效消除这种不匹配。此改变简单明了，现代框架完全支持，仅需少量代码更改，且不需要修改模型架构或学习算法。我们的结果表明，使用FP16通常能带来更稳定的优化、更快的收敛速度和在多种任务、算法和框架中更强的性能。我们希望这些发现能促使对RL微调中精度权衡进行更广泛的重新考虑。",
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
        "total_votes": 19,
        "visits_count": {
          "all": 547,
          "last_7_days": 547
        },
        "public_total_votes": 66
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
      "id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "paper_group_id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "title": "Context Engineering 2.0: The Context of Context Engineering",
      "abstract": "卡尔·马克思曾写道：“人类本质是社会关系的总和”，这表明个体并不是孤立的存在，而是根本上受到与其他个体互动的影响，其中上下文发挥着构成和本质的作用。随着计算机和人工智能的出现，这些上下文不再仅限于纯粹的人际互动：人机互动也被纳入其中。由此，一个中心问题浮现：机器如何能更好地理解我们的情境和目的？为了应对这一挑战，研究人员最近引入了上下文工程的概念。尽管它常被视为代理时代的一项最新创新，但我们认为相关的实践可以追溯到二十多年前。自1990年代初以来，这一领域经历了不同的历史阶段，每个阶段都受到机器智能水平的影响：从围绕原始计算机构建的早期人机互动框架，到今天由智能代理驱动的人机互动范式，未来可能发展到人类水平或超人类智能。在本文中，我们界定上下文工程，提供系统的定义，概述其历史和概念背景，并探讨实践中的关键设计考虑。通过解决这些问题，我们旨在为上下文工程提供概念基础，并勾勒其光明的未来。本文是向更广泛的社区努力进行系统化上下文工程在人工智能系统中应用的一个起点。",
      "paper_summary": null,
      "image_url": "image/2510.26493v1.png",
      "universal_paper_id": "2510.26493",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 206,
          "last_7_days": 206
        },
        "public_total_votes": 17
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
      "id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "paper_group_id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "title": "The Era of Agentic Organization: Learning to Organize with Language Models",
      "abstract": "我们设想一个新的人工智能时代，称为代理组织，其中代理通过协作和并行工作来解决复杂问题，实现超越个体智能的结果。为了实现这一愿景，我们提出了异步思维（AsyncThink）作为与大语言模型进行推理的新范式，它将内部思维过程组织为可并行执行的结构。具体而言，我们提出了一种思维协议，其中组织者动态分配子查询给工人，合并中间知识，并生成连贯的解决方案。更重要的是，这种协议中的思维结构可以通过强化学习进一步优化。实验表明，与并行思维相比，AsyncThink的推理延迟降低了28%，同时提高了数学推理的准确性。此外，AsyncThink能够泛化其学习的异步思维能力，有效应对未见过的任务，而无需额外训练。",
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
          "all": 261,
          "last_7_days": 261
        },
        "public_total_votes": 28
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
      "id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "paper_group_id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "title": "Scaling Latent Reasoning via Looped Language Models",
      "abstract": "现代大型语言模型（LLM）的训练主要通过显式文本生成来进行“思考”，例如思维链（CoT），这将推理推迟到训练后，并未充分利用预训练数据。我们提出并开源了Ouro，取自递归的乌鲁博罗斯，属于一类预训练的循环语言模型（LoopLM），它通过以下方式将推理构建入预训练阶段：(i) 在潜在空间中进行迭代计算，(ii) 使用熵正则化目标来学习深度分配，(iii) 扩展至7.7万亿个标记。Ouro 1.4B和2.6B模型在各类基准测试中表现出色，达到了最高可与12B SOTA LLMs匹敌的结果。通过对照实验，我们表明这种优势并非源自增加的知识容量，而是源于更优的知识处理能力。我们还展示了LoopLM在推理轨迹上与最终输出的对齐程度优于显式CoT。我们希望我们的结果能够展示LoopLM作为推理时代一种新颖的扩展方向的潜力。我们的模型可以在：此http URL中找到。",
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
        "total_votes": 22,
        "visits_count": {
          "all": 873,
          "last_7_days": 873
        },
        "public_total_votes": 78
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
      "id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "paper_group_id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "title": "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning",
      "abstract": "大型语言模型（LLMs）在需要多步骤推理的问题上往往表现不佳。对于小规模的开源模型，带可验证奖励的强化学习（RLVR）在多次尝试后即便正确解答很少被抽取时也会失败，而监督微调（SFT）则倾向于通过严格的逐个模仿长示范进行过拟合。为了填补这个空白，我们提出了监督强化学习（SRL），一个将问题解决重新定义为生成一系列逻辑“动作”的框架。SRL训练模型在每次执行动作之前生成内部推理的独白。它基于模型的动作与从SFT数据集中提取的专家动作之间的相似性逐步提供更平滑的奖励。这样的监督方式即使在所有展开都不正确时也能提供更丰富的学习信号，同时鼓励灵活的推理，以专家的示范为指导。因此，SRL使小型模型能够学习以前无法通过SFT或RLVR学习的复杂问题。此外，使用SRL初始化训练后再通过RLVR进行精细调整，能够获得最佳整体性能。在推理基准之外，SRL在代理软件工程任务中也表现出良好的推广能力，确立了它作为一个强大且多元的训练框架，适用于以推理为导向的LLMs。",
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
        "total_votes": 6,
        "visits_count": {
          "all": 387,
          "last_7_days": 387
        },
        "public_total_votes": 38
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
      "id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "paper_group_id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "title": "Continuous Autoregressive Language Models",
      "abstract": "大型语言模型（LLMs）的效率在根本上受到其逐步、逐个标记生成过程的限制。我们认为，克服这一瓶颈需要为LLM扩展提供一个新的设计维度：增加每个生成步骤的语义带宽。为此，我们引入了连续自回归语言模型（CALM），这是一个从离散的下一个标记预测转向连续的下一个向量预测的范式转变。CALM利用高保真度的自编码器将一段K个标记压缩为一个连续向量，从中可以以超过99.9%的准确率重建原始标记。这使我们能够将语言建模为一系列连续向量，而不是离散标记，从而将生成步骤的数量减少了K倍。这一范式转变需要一个新的建模工具包，因此我们开发了一个全面的无似然框架，以便在连续领域中实现稳健的训练、评估和可控采样。实验表明，CALM显著改善了性能与计算的权衡，在显著较低的计算成本下达到了强离散基线的性能。更重要的是，这些发现确立了下一个向量预测作为通向超高效语言模型的一条强大且可扩展的路径。代码：这个https URL。项目：这个https URL。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 63,
          "last_7_days": 63
        },
        "public_total_votes": 7
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
      "id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "paper_group_id": "019a3a21-a8fc-7107-a577-1dfa3427e6f7",
      "title": "Deep sequence models tend to memorize geometrically; it is unclear why",
      "abstract": "在序列建模中，原子事实的参数记忆主要被抽象为对实体之间共现的穷举查找。我们将这种联想观点与存储记忆的几何观点进行对比。我们首先隔离一个干净且可分析的Transformer推理实例，这与将记忆严格视为训练期间指定的局部共现的存储方式不兼容。相反，模型必须以某种方式综合了自己的原子事实几何，编码了所有实体之间的全球关系，包括不共现的实体。这反过来简化了涉及$\\ell$次组合的困难推理任务，转变成一个易于学习的一步几何任务。\n\n从这一现象中，我们提取出难以解释的神经嵌入几何的基本方面。我们认为，尽管这种几何的出现是基于对局部关联的优化，但不能简单归因于典型的架构或优化压力。反直觉的是，即使这种几何并不比对关联的穷举查找更简洁，它仍然可以学习到优雅的几何。\n\n接着，通过分析与Node2Vec的联系，我们展示了这几何来自一种谱偏差——与现有理论相反，它确实是自然产生的，尽管缺乏各种压力。这一分析也为从业者指明了让Transformer记忆更具几何性的可行空间。我们希望参数记忆的几何视角能鼓励人们重新审视指导研究人员在知识获取、能力、发现和遗忘等领域的默认直觉。",
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
        "total_votes": 6,
        "visits_count": {
          "all": 177,
          "last_7_days": 177
        },
        "public_total_votes": 27
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
      "abstract": "我们推出了Tongyi DeepResearch，这是一种具有自主性的语言模型，专门为长期、深入的信息搜索研究任务而设计。为了激励自主的深入研究能力，Tongyi DeepResearch通过一个端到端的训练框架开发，该框架结合了自主性中期训练和自主性后期训练，使得在复杂任务中实现可扩展的推理和信息搜索成为可能。我们设计了一种高度可扩展的数据合成管道，完全自动化，不依赖于昂贵的人类标注，支持所有训练阶段。通过为每个阶段构建定制化环境，我们的系统实现了稳定且一致的互动。Tongyi DeepResearch拥有305亿个总参数，其中每个token仅激活33亿个参数，在一系列自主深度研究基准测试中实现了最先进的性能，包括人类的最后考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES和xbench-DeepSearch-2510。我们将该模型、框架和完整解决方案开源，以赋能社区。",
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
        "total_votes": 52,
        "visits_count": {
          "all": 1790,
          "last_7_days": 1790
        },
        "public_total_votes": 147
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
      "id": "019a390a-6398-7cfc-8161-edd680691708",
      "paper_group_id": "019a390a-6398-7cfc-8161-edd680691708",
      "title": "The End of Manual Decoding: Towards Truly End-to-End Language Models",
      "abstract": "“大端到端”标签对于大语言模型（LLM）是一个误称。实际上，它们依赖于一个不可微分的解码过程，需要繁琐的手动调整超参数，如温度和top-p。本文介绍了AutoDeco，这是一种新颖的架构，通过学习控制自身的解码策略，实现真正的“端到端”生成。我们在标准变换器上增加了轻量级的头部，在每一步动态预测上下文特定的温度和top-p值，连同下一个token的logits。这种方法将解码转化为一个参数化的token级过程，使模型能够在单次前向传递中自我调节其采样策略。\n\n通过在八个基准上的广泛实验，我们证明了AutoDeco不仅显著超越了默认的解码策略，还达到了与基于“破解测试集”获得的oracle调优基线相当的性能——这是任何静态方法的实际上限。重要的是，我们发现了一种基于指令的解码控制的突现能力：模型学习理解自然语言命令（例如，“以低随机性生成”），并在逐个token的基础上调整其预测的温度和top-p，为可引导和互动的LLM解码开辟了新范式。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 151,
          "last_7_days": 151
        },
        "public_total_votes": 24
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
      "id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "paper_group_id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "title": "ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning",
      "abstract": "多模态推理需要语言和视觉之间的迭代协调，但目前尚不清楚什么构成有意义的交织思维链。我们认为文本和图像思维应作为互补的，而不是同构的模态，互相推动推理。在这一原则的指导下，我们构建了ThinkMorph，一个经过微调的统一模型，基于24,000个高质量的交织推理轨迹，涵盖了不同视觉参与度的任务。ThinkMorph学习生成逐步的文本-图像推理步骤，具体操作视觉内容，同时保持连贯的语言逻辑。它在以视觉为中心的基准测试中表现出显著提升（平均比基础模型高34.7%），并能泛化到领域外任务，匹敌或超越更大和专有的视觉语言模型。除了性能，ThinkMorph还展现了新兴的多模态智能，包括未见过的视觉操作技能、在推理模式之间的自适应切换，以及通过多样化的多模态改进测试时的扩展能力。这些发现为表征统一模型在多模态推理中的新兴能力提供了有前景的方向。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 48,
          "last_7_days": 48
        },
        "public_total_votes": 7
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
      "id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "paper_group_id": "019a390b-4669-7dae-925a-1140f1643bf9",
      "title": "Encoder-Decoder or Decoder-Only? Revisiting Encoder-Decoder Large Language Model",
      "abstract": "最近的大型语言模型（LLM）研究经历了从编码器-解码器建模到如今主导的仅解码器建模的架构转变。然而，这一快速转变缺乏严格的比较分析，尤其是\\textit{从规模的角度}，引发了人们对编码器-解码器模型潜力可能被忽视的担忧。为了填补这一空白，我们重新审视了编码器-解码器LLM（RedLLM），并结合了来自仅解码器LLM（DecLLM）的最新方法。我们在不同的模型规模下，比较了使用前缀语言建模（LM）进行预训练的RedLLM与使用因果LM进行预训练的DecLLM，规模范围从约150M到约8B。采用RedPajama V1（1.6T tokens）进行预训练，并使用FLAN进行指令调优，我们的实验表明，RedLLM展现出引人注目的规模特性和惊人的强大性能。尽管在预训练阶段，DecLLM在计算效率上整体更优，但RedLLM展示了相当的规模和上下文长度外推能力。在指令调优后，RedLLM在多种下游任务上取得了可比甚至更好的结果，同时享有显著更好的推理效率。我们希望我们的发现能够激励更多的研究重新审视RedLLM，挖掘其开发强大且高效的LLM的潜力。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 79,
          "last_7_days": 79
        },
        "public_total_votes": 14
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
      "abstract": "在本文中，我们展示了如何使用单个消费者级GPU以30Hz的帧率和最多480Hz的轨迹频率运行pi0级多视角VLA。这使得以前认为大型VLA模型无法实现的动态实时任务成为可能。为此，我们引入了一系列策略，以消除模型推理中的开销。实际实验表明，采用我们策略的pi0策略在抓取掉落笔的任务中取得了100%的成功率。基于这些结果，我们进一步提出了一个全流式推理框架，用于VLA的实时机器人控制。代码可在此链接获取。",
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
          "all": 94,
          "last_7_days": 94
        },
        "public_total_votes": 17
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
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我提升系统需要与环境互动以实现持续适应。我们引入了SPICE（自我游戏在语料库环境中），这是一个强化学习框架，其中一个模型扮演两个角色：挑战者从大型语料库中提取文档以生成多样化的推理任务，而推理者则解决这些任务。通过对抗性动态，挑战者在推理者能力的边界上创建了自动课程，而语料库的基础提供了持续改进所需的丰富、几乎用之不竭的外部信号。与现有的无基础自我游戏方法相比，后者提供的好处更为有限，SPICE在多个模型系列的数学（+8.9%）和一般推理（+9.8%）基准测试中取得了一致的进步。我们的分析揭示了文档基础是SPICE的关键成分，它能够持续生成自己越来越具有挑战性的目标并实现这些目标，从而实现持续的自我提升。",
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
        "total_votes": 10,
        "visits_count": {
          "all": 351,
          "last_7_days": 351
        },
        "public_total_votes": 49
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
      "id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "paper_group_id": "019a3841-c010-7c80-8d07-18a2a86bde4a",
      "title": "$π_\\texttt{RL}$: Online RL Fine-tuning for Flow-based Vision-Language-Action Models",
      "abstract": "视觉语言行动（VLA）模型使机器人能够理解并执行复杂任务，通过多模态输入。尽管最近的工作探索了使用强化学习（RL）来自动化在扩展监督微调（SFT）中繁琐的数据收集过程，但由于迭代去噪所产生的不可处理的动作对数几率，将大规模 RL 应用于基于流的 VLA（例如，$\\pi_0$，$\\pi_{0.5}$）仍然具有挑战性。我们通过 $\\pi_{\\text{RL}}$ 来解决这个问题，这是一个用于并行仿真训练基于流的 VLA 的开源框架。$\\pi_{\\text{RL}}$ 实现了两种 RL 算法：（1）{Flow-Noise} 将去噪过程建模为具有可学习噪声网络的离散时间马尔可夫决策过程（MDP），以进行精确的对数几率计算。（2）{Flow-SDE} 将去噪与智能体-环境交互结合起来，构建了一个双层 MDP，采用 ODE 到 SDE 转换以实现高效的 RL 探索。我们在 LIBERO 和 ManiSkill 基准上评估了 $\\pi_{\\text{RL}}$。在 LIBERO 上，$\\pi_{\\text{RL}}$ 使少量样本 SFT 模型 $\\pi_0$ 和 $\\pi_{0.5}$ 的表现分别从 57.6% 提升至 97.6% 和从 77.1% 提升至 98.3%。在 ManiSkill 中，我们在 320 个并行环境中训练 $\\pi_{\\text{RL}}$，使 $\\pi_0$ 从 41.6% 提升至 85.7%，$\\pi_{0.5}$ 从 40.0% 提升至 84.8%，涵盖 4352 个抓取与放置任务，展示了在异构仿真下可扩展的多任务 RL。总体而言，$\\pi_{\\text{RL}}$ 在性能上显著提升，并且在 SFT 模型上表现出更强的泛化能力，验证了在线 RL 在基于流的 VLA 中的有效性。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 90,
          "last_7_days": 90
        },
        "public_total_votes": 14
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
      "abstract": "基于大型语言模型(LLM)的网页代理在信息搜索方面展现出巨大的潜力，但在长时任务上的有效性受到一种基本的上下文管理权衡的制约。现有的ReAct基础代理在累积噪声和原始历史时遭遇上下文饱和，而固定地在每一步总结完整历史的方法则面临着关键细节不可逆转的损失。为了解决这些问题，我们引入了AgentFold，这是一种以主动上下文管理为中心的新型代理范式，灵感来源于人类认知过程中的回顾性整合。AgentFold将其上下文视为一个动态的认知工作空间，主动进行塑造，而不是一个被动的日志。在每一步，它学习执行“折叠”操作，以多种尺度管理其历史轨迹：它可以进行细致的浓缩以保留重要的细微细节，或进行深层整合以抽象出整个多步骤子任务。著名基准测试的结果令人瞩目：通过简单的监督微调（不需要持续预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上达到了36.2%，在BrowseComp-ZH上达到了47.3%。值得注意的是，这一表现不仅超越或匹配了规模大得多的开源模型，如DeepSeek-V3.1-671B-A37B，还超越了领先的专有代理，如OpenAI的o4-mini。",
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
          "all": 407,
          "last_7_days": 407
        },
        "public_total_votes": 53
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
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLMs）推理能力方面展示了显著的潜力。然而，RL在LLMs中的成功在很大程度上依赖于人工策划的数据集和可验证的奖励，这限制了其可扩展性和普适性。最近的自我对弈RL方法受到游戏和围棋领域成功的启发，旨在在没有人工标注数据的情况下提升LLM的推理能力。然而，它们的方法主要依赖于一个实际环境提供反馈（例如，Python解释器或游戏引擎）；将其扩展到通用领域仍然具有挑战性。为了解决这些问题，我们提出了多智能体进化（MAE）框架，它使LLM能够在解决多样任务时自我进化，包括数学、推理和一般知识问答。MAE的核心设计基于一组三个相互作用的代理（提问者、求解者、评估者），它们从一个单一的LLM实例化，并应用强化学习优化其行为。提问者生成问题，求解者尝试解决方案，评估者在共同进化的过程中评估两者。针对Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试上实现了平均提高4.54%的成绩。这些结果凸显了MAE作为一种可扩展的、数据高效的方法，可以在最小依赖于人工策划监督的情况下提升LLM的普遍推理能力。",
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
        "total_votes": 19,
        "visits_count": {
          "all": 802,
          "last_7_days": 802
        },
        "public_total_votes": 81
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
      "id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "paper_group_id": "019a382e-004c-7d31-a82d-ebbd8d4ee68b",
      "title": "Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-CoF Benchmark",
      "abstract": "最近的视频生成模型能够生成高保真、时序一致的视频，这表明它们可能编码了大量的世界知识。除了现实合成之外，它们还表现出一些新兴行为，表明它们具有视觉感知、建模和操控的能力。然而，一个重要问题仍然存在：在具有挑战的视觉推理场景中，视频模型是否准备好作为零样本推理器？在本研究中，我们进行了一项实证研究，以全面调查这个问题，重点关注领先且受欢迎的Veo-3。我们评估其在12个维度上的推理行为，包括空间、几何、物理、时间和具身逻辑，系统地描绘了其优点和失败模式。为了标准化这项研究，我们将评估数据整理为MME-CoF，这是一个紧凑的基准，能够深入且全面地评估帧链（Chain-of-Frame, CoF）推理。我们的研究结果表明，尽管目前的视频模型在短时空间一致性、细粒度基础和局部一致动态方面展现出有希望的推理模式，但在长时因果推理、严格的几何约束和抽象逻辑方面仍然有限。总体而言，它们尚不可靠作为独立的零样本推理器，但作为专门推理模型的补充视觉引擎展现出令人鼓舞的迹象。项目页面：此链接。",
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
          "all": 74,
          "last_7_days": 74
        },
        "public_total_votes": 16
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
      "id": "019a4782-1d96-7c98-8bc1-843eaa2fdb30",
      "paper_group_id": "019a4782-1d96-7c98-8bc1-843eaa2fdb30",
      "title": "Chain-of-Thought Hijacking",
      "abstract": "大型推理模型（LRMs）通过分配更多的推理时间计算来实现更高的任务表现，而先前的研究表明，这种规模化推理也可能通过改进拒绝机制来增强安全性。然而我们发现相反的情况：相同的推理可以用来绕过安全保护。我们引入了链式思维劫持，这是对推理模型的一种越狱攻击。该攻击通过用长序列的无害难题推理来填充有害请求。在HarmBench测试中，链式思维劫持在Gemini 2.5 Pro、GPT o4 mini、Grok 3 mini和Claude 4 Sonnet上的攻击成功率分别达到99%、94%、100%和94%——远远超过了以前的LRMs越狱方法。为了理解我们攻击的有效性，我们进行了机械分析，显示中间层编码了安全检查的强度，而后期层编码了验证结果。长时间的无害链式思维通过将注意力转移 away 有害符号来稀释这两种信号。根据这种分析确定的注意头的有针对性消融实验证明了它们在安全子网络中的作用，确实降低了拒绝。这些结果表明，当显式链式思维与最终答案线索相结合时，最易解释的推理形式本身可以成为一种越狱向量。我们发布了提示、输出和评判决定，以便于复制。",
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
          "all": 23,
          "last_7_days": 23
        },
        "public_total_votes": 4
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
      "id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "paper_group_id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "title": "Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks",
      "abstract": "人类拥有空间推理能力，使他们能够通过多模态观察（例如视觉和声音）理解空间。大型多模态推理模型通过学习感知和推理来扩展这些能力，在各种空间任务中展现出良好的表现。然而，针对这些模型的系统评价和公开可用的基准仍然有限。在本次调查中，我们对大型模型的多模态空间推理任务进行了全面回顾，分类了多模态大语言模型（MLLMs）的最新进展，并介绍了评估的开放基准。我们首先概述一般的空间推理，重点讨论后期训练技术、可解释性和架构。除了经典的二维任务外，我们还考察了空间关系推理、场景和布局理解，以及视觉问答和在三维空间中的基础。我们还回顾了在具身人工智能方面的进展，包括视觉-语言导航和动作模型。此外，我们还考虑了新兴的模态，如音频和自我中心视频，这些通过新传感器为新颖的空间理解提供支持。我们相信这项调查建立了一个坚实的基础，并为不断发展中的多模态空间推理领域提供了洞见。有关本次调查的最新信息、代码和开放基准的实现可以在此 https URL 找到。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 119,
          "last_7_days": 119
        },
        "public_total_votes": 24
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
      "id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "paper_group_id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "title": "GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning",
      "abstract": "由大型语言模型（LLMs）驱动的自主代理在复杂任务解决中的工具操作能力表现出色。然而，现有的范式如ReAct依赖于顺序推理和执行，未能利用独立子任务之间的固有并行性。这一顺序瓶颈导致了工具利用效率低下和多步推理场景中表现不佳。我们提出了基于图的代理规划（GAP），这是一个新颖的框架，通过基于图的规划显式建模任务间的依赖关系，以实现自适应的并行和串行工具执行。我们的方法训练代理基础模型，将复杂任务分解为关注依赖关系的子任务图，自主确定哪些工具可以并行执行，哪些必须遵循顺序依赖。这种关注依赖关系的协调在执行效率和任务准确性上都取得了显著提高。为了训练GAP，我们构建了一个高质量的数据集，该数据集来源于多跳问答（MHQA）基准的基于图的规划轨迹。我们采用两阶段的训练策略：首先在整理好的数据集上进行监督微调（SFT），然后在战略性抽样的查询上使用基于正确性的奖励函数进行强化学习（RL），这些查询中基于工具的推理提供了最大价值。MHQA数据集上的实验结果表明，GAP的表现显著优于传统的ReAct基线，特别是在多步检索任务中，同时通过智能并行化实现了工具调用效率的显著提升。项目页面可在：这个https URL上访问。",
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
          "all": 96,
          "last_7_days": 96
        },
        "public_total_votes": 14
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
      "id": "019a4816-97ae-711f-8c4a-c6055c39d3d6",
      "paper_group_id": "019a4816-97ae-711f-8c4a-c6055c39d3d6",
      "title": "Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning",
      "abstract": "空间理解仍然是大型视觉语言模型（LVLMs）的一个薄弱环节。现有的监督微调（SFT）和最近的可验证奖励强化学习（RLVR）流程依赖于昂贵的监督、专业的工具或受限的环境，这限制了其规模。我们推出了空间自监督强化学习（Spatial-SSRL），这一自监督RL范式直接从普通的RGB或RGB-D图像中提取可验证信号。Spatial-SSRL自动制定了五个前置任务，以捕捉二维和三维空间结构：打乱的补丁重新排序、翻转的补丁识别、裁剪的补丁修复、区域深度排序和相对三维位置预测。这些任务提供了易于验证的真实答案，并且无需人类或LVLM的注释。在我们的任务上训练显著增强了空间推理能力，同时保留了基本的视觉能力。在七个空间理解基准测试中，无论是在图像还是视频设置下，Spatial-SSRL在Qwen2.5-VL基准上提供了平均准确率的提升，分别为4.63%（3B）和3.89%（7B）。我们的结果表明，简单的内在监督使得RLVR能够大规模应用，并为LVLMs提供了增强空间智能的实用途径。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 14,
          "last_7_days": 14
        },
        "public_total_votes": 3
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
      "id": "019a342f-15a1-7a07-86c2-7c19f79d99aa",
      "paper_group_id": "019a342f-15a1-7a07-86c2-7c19f79d99aa",
      "title": "INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats",
      "abstract": "现代AI硬件，比如Nvidia的Blackwell架构，正越来越多地采用低精度浮点（FP）格式来处理大型语言模型（LLMs）中普遍存在的激活异常值。尽管这一行业趋势明显，但针对不同细粒度的FP和整数（INT）量化的统一比较仍然缺失，这使得算法与硬件的协同设计缺乏明确指导。本文填补了这一空白，通过系统地研究FP和INT格式之间的权衡。我们揭示了一个关键的性能交叉点：虽然FP在粗粒度量化中表现出色，但在细粒度（块级）层面的比较更为复杂。我们的综合比较表明，对于流行的8位细粒度格式（例如，区块大小为32的MX），MXINT8在算法准确性和硬件效率上均优于其FP对应物。然而，对于4位格式，FP（例如，MXFP4、NVFP4）通常具有准确性优势，尽管我们展示了在应用哈达玛旋转等异常值缓解技术时，NVINT4可以超越NVFP4。我们还提出了一种对称裁剪方法，解决了细粒度低比特INT训练中的梯度偏差，从而使MXINT8训练几乎无损地实现性能。我们的发现对当前硬件发展趋势提出了挑战，表明单一的FP方法并非最佳选择，并倡导细粒度INT格式，特别是MXINT8，为未来的AI加速器提供了更好的准确性、功耗和效率平衡。",
      "paper_summary": {
        "summary": "A comprehensive study on fine-grained low-bit quantization formats reveals that integer (INT) formats can outperform floating-point (FP) in both algorithmic accuracy and hardware efficiency for large language models, particularly MXINT8, which achieves nearly lossless training and reduces hardware energy by 37% and area by 21% compared to MXFP8.",
        "originalProblem": [
          "The exponential growth of Large Language Models (LLMs) creates immense computational and memory demands, necessitating aggressive quantization for efficient deployment.",
          "Traditional low-precision quantization struggles with activation outliers in LLMs, leading the AI hardware industry to increasingly adopt low-precision floating-point (FP) formats based on assumptions about dynamic range.",
          "A systematic, unified comparison of FP and INT quantization across varying granularities, especially fine-grained (block-wise) levels, was lacking, leaving critical gaps for optimal algorithm and hardware co-design."
        ],
        "solution": [
          "A theoretical framework based on Quantization Signal-to-Noise Ratio (QSNR) was developed to compare INT and FP formats, identifying crossover points based on data characteristics like crest factor.",
          "Extensive empirical validation was performed on 12 diverse LLMs for both direct-cast inference and low-bit training, evaluating INT and FP variants of MX and NV formats across 8-bit, 6-bit, and 4-bit precisions.",
          "A gate-level hardware model for Matrix-Multiply Units was used to conduct a detailed cost analysis, quantifying the area and energy efficiency of fine-grained INT formats compared to their FP counterparts."
        ],
        "keyInsights": [
          "A crucial performance crossover point exists, where INT formats become superior to FP formats when quantization granularity is fine-grained, effectively reducing the data's crest factor.",
          "MXINT8 consistently demonstrates higher algorithmic accuracy and substantially greater hardware efficiency (37% less energy, 21% less area) than MXFP8, enabling nearly lossless LLM training.",
          "While 4-bit FP often holds an initial advantage, 4-bit INT formats like NVINT4 can surpass NVFP4 in accuracy when combined with effective outlier-mitigation techniques such as Hadamard rotation."
        ],
        "results": [
          "MXINT8 achieved an average QSNR of 40.35 dB, significantly higher than MXFP8's 31.50 dB, and outperformed MXFP8 across all 12 evaluated LLMs in direct-cast inference.",
          "Hardware analysis revealed that MXINT8 reduced energy consumption by 37% and chip area by 21% compared to MXFP8, while its training performance was nearly indistinguishable from BFloat16.",
          "With random Hadamard rotation, NVINT4 surpassed NVFP4 in direct-cast inference across all 12 LLMs, showing a 99.3% win rate in tensor-wise QSNR comparisons."
        ]
      },
      "image_url": "image/2510.25602v1.png",
      "universal_paper_id": "2510.25602",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 56,
          "last_7_days": 56
        },
        "public_total_votes": 15
      },
      "first_publication_date": "2025-10-29T15:11:53.000Z",
      "publication_date": "2025-10-29T15:11:53.000Z",
      "updated_at": "2025-10-30T08:14:52.321Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "energy-efficient-ml",
        "hardware-aware-algorithms",
        "inference-optimization",
        "lightweight-models",
        "ml-systems",
        "model-compression",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 7,
      "github_url": "https://github.com/ChenMnZ/INT_vs_FP",
      "distance": 1
    },
    {
      "id": "019a3a44-561c-793b-addd-93a35fdb5a05",
      "paper_group_id": "019a3a44-561c-793b-addd-93a35fdb5a05",
      "title": "From Embedding to Control: Representations for Stochastic Multi-Object Systems",
      "abstract": "本文研究如何在具有多个交互对象的随机非线性动态中实现准确建模和有效控制。然而，非均匀交互和随机拓扑使这一任务具有挑战性。我们通过提出\\textit{图可控嵌入}（GCE）来解决这些挑战，这是一种用于学习线性控制的随机多对象动态的通用框架。具体而言，GCE建立在希尔伯特空间嵌入的基础上，允许将受控随机动态的概率分布直接嵌入到再生核希尔伯特空间（RKHS）中，这使得它在RKHS中能够进行线性操作，同时保留非线性表达能力。我们提供了关于GCE存在性、收敛性和适用性的理论保证。值得注意的是，采取了均值场近似技术来高效捕捉对象间依赖关系，并实现可证明的低样本复杂度。通过整合图神经网络，我们构建了数据依赖的核特征，能够适应动态交互模式，并在仅有有限训练实例的情况下推广到未见过的拓扑。GCE能够无缝扩展到不同大小和拓扑的多对象系统。利用希尔伯特空间的线性特性，GCE还支持简单但有效的控制算法，用于合成最佳序列。在物理系统、机器人技术和电力网的实验验证了GCE，并在分布内和少量样本测试中表现出比各种竞争性嵌入方法更为一致的性能提升。",
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
          "all": 46,
          "last_7_days": 46
        },
        "public_total_votes": 10
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
    },
    {
      "id": "019a3fd5-f17a-78da-b7e2-a84569676df2",
      "paper_group_id": "019a3fd5-f17a-78da-b7e2-a84569676df2",
      "title": "Accelerating mathematical research with language models: A case study of an interaction with GPT-5-Pro on a convex analysis problem",
      "abstract": "最近，大型语言模型在数学领域的进展使它们越来越能够充当研究助手。然而，随着它们推理能力的提升，评估它们的数学能力变得越来越具挑战性。用于评估的问题必须既不能太简单也不能太困难，单一的数值评分已无法概括它们的表现，而有意义的评估需要专家的监督。\n\n在这项工作中，我们研究了作者与大型语言模型之间的互动，以证明一个关于凸优化的引理。具体而言，我们建立了双共轭算子的梯度的泰勒展开——即通过两次应用Fenchel变换获得的算子——围绕一个严格凸函数进行，并得到了OpenAI最新模型GPT-5-pro的帮助。\n\n除了数学结果本身外，我们并不确切声称其新颖性，我们的主要贡献在于记录这种协作推理过程。GPT-5-pro通过建议相关的研究方向和证明一些中间结果加速了我们的进展。然而，它的推理仍然需要仔细的监督，特别是为了纠正细微的错误。尽管这一实验仅限于单个数学问题和单一语言模型，但它展示了大型语言模型作为数学合作者的潜力与现有限制。",
      "paper_summary": {
        "summary": "A qualitative case study documented the collaborative reasoning process between a human mathematician and GPT-5-Pro on a convex analysis problem, leading to the rigorous proof of a first-order Taylor expansion for the gradient of the biconjugation operator. The language model accelerated the research by generating key conjectures and intermediate proof steps, while human supervision was crucial for correcting subtle logical errors.",
        "originalProblem": [
          "To rigorously prove a specific variant of a convex analysis lemma (Equation 2 in the paper) required for ongoing optimal transport research.",
          "This particular variant, concerning the Taylor expansion of the biconjugation operator's gradient, was not readily available in existing literature.",
          "To qualitatively evaluate the capabilities and limitations of GPT-5-Pro as a collaborative research assistant in advanced mathematical problem-solving."
        ],
        "solution": [
          "The author engaged in an iterative, meticulously documented dialogue with GPT-5-Pro, providing initial queries and then continuously guiding, challenging, and correcting the LLM's proof attempts.",
          "The approach involved leveraging standard convex analysis tools such as the Fenchel transform, biconjugation, subdifferentials, and Alexandrov's theorem.",
          "The final proof strategy, suggested by the author, focused on characterizing the biconjugate as an affine envelope and demonstrating a suitable affine minorant."
        ],
        "keyInsights": [
          "GPT-5-Pro significantly accelerated the research process by suggesting crucial conjectures (e.g., that the $o(t)$ remainder term is zero) and generating intermediate proof segments.",
          "Even advanced LLMs require careful human supervision to detect and correct subtle logical errors, misapplications of theorems, and incorrect assumptions during complex mathematical reasoning.",
          "Qualitative evaluation of LLMs in realistic research scenarios provides valuable insights into their \"reasoning\" capabilities and limitations beyond traditional automated benchmarks."
        ],
        "results": [
          "A rigorous proof was successfully established for the first-order Taylor expansion of the gradient of the biconjugation operator (Proposition 1 and 2), demonstrating that the $o(t)$ remainder term is exactly zero for sufficiently small $|t|$.",
          "GPT-5-Pro successfully identified a key conjecture that simplified the problem and contributed relevant research directions and partially correct proof segments.",
          "The collaboration highlighted GPT-5-Pro's tendencies for errors in applying strong convexity assumptions, bounding terms in inequalities, and confusing local properties with global ones, underscoring the continued necessity of human mathematical expertise."
        ]
      },
      "image_url": "image/2510.26647v1.png",
      "universal_paper_id": "2510.26647",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 24,
          "last_7_days": 24
        },
        "public_total_votes": 7
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
      "id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "paper_group_id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "title": "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender",
      "abstract": "在推荐系统中，扩大特征交互模块（如 Wukong、RankMixer）或用户行为序列模块（如 LONGER）已取得显著成功。然而，这些努力通常在不同的轨道上进行，这不仅阻碍了双向信息交换，还妨碍了统一优化和扩展。本文提出了 OneTrans，这是一种统一的 Transformer 主干，能够同时执行用户行为序列建模和特征交互。OneTrans 采用统一的分词器将顺序和非顺序属性转换为单一的令牌序列。堆叠的 OneTrans 模块在相似的顺序令牌之间共享参数，同时为非顺序令牌分配特定的参数。通过因果注意力和跨请求的 KV 缓存，OneTrans 使得中间表示的预计算和缓存成为可能，显著降低了训练和推理期间的计算成本。在工业规模数据集上的实验结果表明，OneTrans 在参数增加时有效扩展，始终优于强基线，并在在线 A/B 测试中实现了每用户 GMV 提升 5.68%。",
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
          "all": 122,
          "last_7_days": 122
        },
        "public_total_votes": 22
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
      "id": "019a48b7-d9c3-7625-97a0-47bac9e7457d",
      "paper_group_id": "019a48b7-d9c3-7625-97a0-47bac9e7457d",
      "title": "Higher-order Linear Attention",
      "abstract": "缩放点积注意力的二次成本是将自回归语言模型扩展到长上下文的核心障碍。线性时间注意力和状态空间模型（SSMs）提供了可扩展的替代方案，但通常仅限于一阶或基于核的近似，这可能限制表达能力。我们介绍了高阶线性注意力（HLA），这是一种因果的、流式的机制，通过紧凑的前缀充分统计量实现更高阶的交互。在二阶情况下，HLA 维持一个大小恒定的状态，并以线性时间计算每个标记的输出，而无需形成任何 $n \\times n$ 矩阵。我们给出了封闭式流式恒等式，使用两个额外摘要的严格因果掩蔽变体，以及一种基于关联扫描的分块并行训练方案，能精确再现串行递归的激活。我们还概述了对三阶及更高阶的扩展。总体来看，这些结果将 HLA 定位为一个有原则的、可扩展的基础构件，结合了类似注意力的、数据依赖的混合和现代循环架构的效率。项目页面：https URL。",
      "paper_summary": null,
      "image_url": "image/2510.27258v1.png",
      "universal_paper_id": "2510.27258",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 11,
          "last_7_days": 11
        },
        "public_total_votes": 3
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
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "paper_group_id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "title": "The Denario project: Deep knowledge AI agents for scientific discovery",
      "abstract": "我们介绍了Denario，这是一个旨在作为科学研究助手的AI多智能体系统。Denario可以执行许多不同的任务，例如生成创意、查阅文献、制定研究计划、编写和执行代码、制作图表以及草拟和审阅科学论文。该系统具有模块化架构，能够处理特定任务，例如生成创意或使用Cmbagent作为深度研究后端进行端到端的科学分析。在这项工作中，我们详细描述了Denario及其模块，并通过展示其在许多不同科学学科（如天体物理学、生物学、生物物理学、生物医学信息学、化学、材料科学、数学物理学、医学、神经科学和行星科学）生成的多篇AI生成论文来说明其能力。Denario还擅长将来自不同学科的想法结合在一起，我们通过展示一篇将量子物理学和机器学习方法应用于天体物理数据的论文来说明这一点。我们报告了领域专家对这些论文的评估，他们提供了数值评分和类似审稿的反馈。接着，我们强调了当前系统的优点、缺点和局限性。最后，我们讨论了AI驱动研究的伦理影响，并反思这种技术与科学哲学的关系。我们将在此HTTPS URL公开发布代码。Denario演示也可以直接在此HTTPS URL上运行，完整应用程序将部署在云端。",
      "paper_summary": null,
      "image_url": "image/2510.26887v1.png",
      "universal_paper_id": "2510.26887",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 11,
          "last_7_days": 11
        },
        "public_total_votes": 4
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
      "id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "paper_group_id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "title": "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents",
      "abstract": "关于大规模监督微调AI代理的公共研究结果相对较少，因为收集代理训练数据面临独特的挑战。在本研究中，我们认为瓶颈并不是缺乏基础数据源，而是大量异构格式、工具和接口中的数据碎片化。为此，我们引入了代理数据协议（ADP），这是一种轻量级的表示语言，充当不同格式的代理数据集与下游统一代理训练管道之间的“中介语言”。ADP的设计足够表达，以捕捉多种任务，包括API/工具使用、浏览、编码、软件工程和一般代理工作流程，同时保持解析和训练的简单性，无需在每个数据集层面进行工程设计。在实验中，我们将13个现有的代理训练数据集统一转换为ADP格式，并将标准化的ADP数据转换为多个代理框架的训练就绪格式。我们对这些数据进行了监督微调，展示了约20%的平均性能提升，相比对应的基础模型，在标准编码、浏览、工具使用和研究基准上实现了最先进或接近最先进的性能，无需领域特定调优。所有代码和数据都已公开发布，希望ADP能够帮助降低标准化、可扩展和可重复的代理训练的门槛。",
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
        "total_votes": 8,
        "visits_count": {
          "all": 314,
          "last_7_days": 314
        },
        "public_total_votes": 41
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
    }
  ],
  "page": 0
};