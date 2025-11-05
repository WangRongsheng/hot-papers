const papersData = {
  "papers": [
    {
      "id": "019a4d9b-eed2-7b77-a036-53431925e9c6",
      "paper_group_id": "019a4d9b-eed2-7b77-a036-53431925e9c6",
      "title": "Towards Robust Mathematical Reasoning",
      "abstract": "找到合适的北极星指标对于提升基础模型的数学推理能力至关重要，尤其是因为现有的评估要么过于简单，要么仅关注获取正确的简短答案。为了解决这些问题，我们推出了IMO-Bench，这是一套经过顶尖专家审查的高级推理基准，专门针对国际数学奥林匹克（IMO）的水平，这是年轻数学家最负盛名的赛事。IMO-AnswerBench首先在400个具有可验证简短答案的多样化奥林匹克问题上测试模型。IMO-Proof Bench是针对证明写作能力的下一层级评估，包括基础和高级IMO水平的问题，以及详细的评分指南以促进自动评分。这些基准在我们2025年以Gemini Deep Think（Luong和Lockhart，2025）实现金级表现的历史成就中发挥了至关重要的作用。我们的模型在IMO-AnswerBench上取得了80.0%的成绩，在高级IMO-Proof Bench上取得了65.7%的成绩，分别超越了最佳的非Gemini模型6.9%和42.4%。我们还表明，使用Gemini推理构建的自动评分器与人工评估高度相关，并构建了IMO-GradingBench，包含1000个证明的人工评分，以推动长文本答案的自动评估的进一步进展。我们希望IMO-Bench能帮助社区提升稳健的数学推理，并在该网址发布。",
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
        "total_votes": 9,
        "visits_count": {
          "all": 200,
          "last_7_days": 200
        },
        "public_total_votes": 24
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
      "id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "paper_group_id": "019a4216-7938-7e0d-85a1-f5efa61f3b81",
      "title": "Context Engineering 2.0: The Context of Context Engineering",
      "abstract": "卡尔·马克思曾写道“人类本质是社会关系的总和”，这表明个体并不是孤立的存在，而是根本上受制于与其他实体的互动，其中文化背景扮演着构成性和本质性的角色。随着计算机和人工智能的出现，这些背景不再仅限于纯粹的人与人之间的互动：人与机器之间的互动也被纳入其中。那么，一个中心问题便随之而来：机器如何能够更好地理解我们的情境和目的？为了解决这一挑战，研究人员最近引入了“情境工程”的概念。尽管它通常被视为代理时代的一项新创新，但我们认为相关实践可以追溯到二十多年前。自20世纪90年代初以来，该领域经历了不同的历史阶段，每个阶段都受到机器智能水平的影响：从围绕原始计算机构建的早期人机交互框架，到如今由智能代理驱动的人机互动范式，未来可能实现人类水平或超人类智能。在本文中，我们将情境工程的相关内容进行定位，提供一个系统的定义，概述其历史和概念框架，并探讨实践中的关键设计考虑。通过回答这些问题，我们旨在为情境工程提供一个概念基础，并描绘其光明的未来。本文是为了推动更广泛社区对AI系统中系统化情境工程的努力而迈出的第一步。",
      "paper_summary": null,
      "image_url": "image/2510.26493v1.png",
      "universal_paper_id": "2510.26493",
      "metrics": {
        "total_votes": 34,
        "visits_count": {
          "all": 1268,
          "last_7_days": 1268
        },
        "public_total_votes": 95
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
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们介绍了Kimi Linear，一种混合线性注意力架构，它首次在各种场景下（包括短期上下文、长期上下文和强化学习（RL）扩展模式）以公平的比较超越了全注意力架构。其核心是Kimi Delta Attention（KDA），一种富有表现力的线性注意力模块，通过更精细的门控机制扩展了Gated DeltaNet，从而更有效地利用有限的有限状态RNN内存。我们定制的分块算法通过一种特殊变体的对角加低秩（DPLR）转移矩阵实现了高硬件效率，相比于一般的DPLR公式，显著减少了计算量，同时与经典的delta规则保持更一致。\n\n我们预训练了一个Kimi Linear模型，具有30亿激活参数和48亿总参数，基于KDA和多头潜在注意力（MLA）的层级混合。我们的实验表明，在相同的训练方案下，Kimi Linear在所有评估任务中表现优于完整的MLA，且其KV缓存使用量减少了多达75%，在1M上下文中实现了高达6倍的解码吞吐量。这些结果表明，Kimi Linear可以作为全注意力架构的替代品，具备更优的性能和效率，包括处理更长的输入和输出长度的任务。\n\n为了支持进一步的研究，我们开源了KDA内核和vLLM实现，并发布了预训练和指令调优的模型检查点。",
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
        "total_votes": 60,
        "visits_count": {
          "all": 2226,
          "last_7_days": 2226
        },
        "public_total_votes": 160
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
      "id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "paper_group_id": "019a47aa-dc16-73db-b33b-ed55797d6c9d",
      "title": "Continuous Autoregressive Language Models",
      "abstract": "大型语言模型（LLMs）的效率基本上受到其逐步、逐个标记生成过程的限制。我们认为，克服这一瓶颈需要为LLM扩展提供新的设计方向：增加每个生成步骤的语义带宽。为此，我们引入了连续自回归语言模型（CALM），这是一种从离散下一个标记预测转变为连续下一个向量预测的范式。CALM使用高保真度的自编码器将一段K个标记压缩为一个单一的连续向量，从中可以以超过99.9%的准确率重构原始标记。这使我们能够将语言建模为一系列连续向量，而不是离散的标记，从而将生成步骤的数量减少了K倍。这一范式转变需要新的建模工具包；因此，我们开发了一个全面的无似然框架，能够在连续领域中实现稳健的训练、评估和可控抽样。实验表明，CALM显著改善了性能与计算成本的权衡，在显著降低计算成本的情况下，实现了强大离散基线的性能。更重要的是，这些发现确立了下一个向量预测作为通向超高效语言模型的强大且可扩展的途径。代码：此 https URL。项目：此 https URL。",
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
        "total_votes": 10,
        "visits_count": {
          "all": 538,
          "last_7_days": 538
        },
        "public_total_votes": 44
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
      "id": "019a4ff4-3f5c-7bd4-9fc1-f842592b436a",
      "paper_group_id": "019a4ff4-3f5c-7bd4-9fc1-f842592b436a",
      "title": "SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding",
      "abstract": "推测解码已成为加速大型语言模型（LLM）推理的标准方法。它利用无损的草拟-验证程序来绕过自回归解码的延迟，实现了显著的加速。然而，当前的推测解码方法依然受到两个基本瓶颈的限制：（1）草拟过程中的自回归依赖性限制了并行性，以及（2）由于草拟模型与验证模型之间的不对齐而导致的草拟令牌的频繁拒绝。本文提出了SpecDiff-2，一个新颖的框架，以共同解决这两个瓶颈。它利用离散扩散作为非自回归草拟器来解决瓶颈（1），并开发了新技术来校准离散扩散草拟器与自回归验证器，以解决瓶颈（2）。在全面的基准测试套件中的实验结果表明，SpecDiff-2在推理、编码和数学基准上达到了新的先进水平，相较于之前的基准，令牌每秒提高了平均+55%，并在标准解码上获得了最高5.5倍的平均加速，而没有任何准确性的损失。",
      "paper_summary": {
        "summary": "Researchers from the University of Virginia introduce SpecDiff-2, a framework that accelerates Large Language Model inference by using non-autoregressive discrete diffusion models for draft generation, combined with novel train-time and test-time alignment strategies. The method achieves an average 4.22x speed-up over vanilla autoregressive decoding and a 55% increase in tokens-per-second compared to prior speculative decoding baselines, all while maintaining the verifier model's original accuracy.",
        "originalProblem": [
          "Existing speculative decoding methods for Large Language Models (LLMs) are bottlenecked by autoregressive drafter models, which generate tokens sequentially and limit true parallelism for long drafts.",
          "Frequent rejections of drafter proposals by the more powerful verifier model, due to misalignment in token distributions, reduce overall speed-up and necessitate repeated verification cycles.",
          "Prior solutions often addressed drafter latency or alignment separately, or sometimes compromised output quality, failing to provide a joint, lossless acceleration approach."
        ],
        "solution": [
          "Utilizes discrete diffusion language models (DLMs) as non-autoregressive drafters, which generate entire multi-token draft sequences in parallel through a fixed number of denoising steps.",
          "Introduces 'streak-distillation' as a train-time alignment mechanism, fine-tuning the diffusion drafter to optimize for long accepted streaks by the verifier, bridging the distributional gap across the entire draft window.",
          "Deploys 'self-selection acceptance' at test-time, sampling multiple candidate drafts from the diffusion model's marginal distributions and using a verifier-derived score to select the most promising draft for verification, maximizing expected throughput."
        ],
        "keyInsights": [
          "Discrete diffusion models offer a powerful non-autoregressive paradigm for speculative decoding drafters, enabling genuine parallel generation of multi-token sequences.",
          "Effective alignment of diffusion drafters with autoregressive verifiers requires optimizing for the entire accepted streak rather than just individual token predictions.",
          "The concept of 'acceleration-compute' shows that investing in faster inference directly translates to higher accuracy on complex reasoning tasks by allowing more reasoning steps within a fixed wall-time budget.",
          "Leveraging the diffusion model's position-wise marginals at inference time allows for efficient generation and selection of multiple high-quality draft candidates."
        ],
        "results": [
          "Achieved an average 4.22x speed-up across diverse benchmarks and verifiers, representing a 30% increase over EAGLE-2, and a +55% improvement in tokens-per-second over previous speculative decoding baselines without any loss of accuracy.",
          "Demonstrated significantly longer average accepted streak lengths (e.g., 5.98 tokens per draft for Qwen2.5-72B at T=0 vs. 4.41 for EAGLE-2), indicating superior drafter-verifier alignment.",
          "On complex reasoning tasks like Math-500 with Chain-of-Thought, acceleration via SpecDiff-2 led to a +63% accuracy boost over the vanilla model and an +11% increase over unaligned SpecDiff within a 15-second reasoning budget, validating the 'acceleration-compute' paradigm.",
          "Ablation studies confirmed that train-time streak-distillation provides a +30% speed-up over base diffusion drafters, and test-time self-selection acceptance yields up to +20% additional throughput, especially at higher temperatures."
        ]
      },
      "image_url": "image/2511.00606v1.png",
      "universal_paper_id": "2511.00606",
      "metrics": {
        "total_votes": 7,
        "visits_count": {
          "all": 120,
          "last_7_days": 120
        },
        "public_total_votes": 17
      },
      "first_publication_date": "2025-11-01T16:12:56.000Z",
      "publication_date": "2025-11-01T16:12:56.000Z",
      "updated_at": "2025-11-04T17:39:58.428Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "efficient-transformers",
        "generative-models",
        "inference-optimization",
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
      "id": "019a4e15-0701-7cea-bed7-5f088f038619",
      "paper_group_id": "019a4e15-0701-7cea-bed7-5f088f038619",
      "title": "Simulating Environments with Reasoning Models for Agent Training",
      "abstract": "LLM 代理在需要深度推理的紧凑环境中表现出色，但在运作于更广泛、更复杂的背景时却依然脆弱，这要求在多样化工具和架构中具备鲁棒性。为训练构建定制环境往往重且脆，限制了进展。在本文中，我们展示了 LLM 可以在没有实际测试数据或 API 访问的情况下模拟真实的环境反馈。受到这种能力的启发，我们提出了两个框架：Simia-SFT，一个通过以环境无关的方式将小种子集放大为多样轨迹来合成 SFT 数据的流程，以及 Simia-RL，一个通过 LLM 模拟反馈实现无真实环境实现的 RL 训练的框架。微调开放模型在多个基准上产生了一致的改进，超越了 GPT-4o，并在 $\\tau^2$-Bench 上接近 o4-mini。Simia-SFT 和 Simia-RL 共同实现了无环境工程的可扩展代理训练，以灵活的 LLM 基础模拟替代重且脆弱的实现。",
      "paper_summary": null,
      "image_url": "image/2511.01824v1.png",
      "universal_paper_id": "2511.01824",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 87,
          "last_7_days": 87
        },
        "public_total_votes": 14
      },
      "first_publication_date": "2025-11-03T18:29:57.000Z",
      "publication_date": "2025-11-03T18:29:57.000Z",
      "updated_at": "2025-11-04T08:56:32.257Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "fine-tuning",
        "generative-models",
        "reasoning",
        "reinforcement-learning",
        "synthetic-data"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 3,
      "github_url": "https://github.com/microsoft/Simia-Agent-Training",
      "distance": 1
    },
    {
      "id": "019a4e0c-8ea4-7da9-92d2-05a68687d4e0",
      "paper_group_id": "019a4e0c-8ea4-7da9-92d2-05a68687d4e0",
      "title": "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI",
      "abstract": "现代聊天机器人大型语言模型生成的“AI助手”角色特征影响着表面行为以及明显的价值观、信念和伦理。这些因素都会影响互动质量、感知智能以及与开发者和用户意图的一致性。塑造这一角色的过程称为角色训练，是业内训练后阶段的关键组成部分，但在学术文献中尚未得到有效研究。我们首次提出角色训练的开放实现，利用宪法AI和一条新的数据管道，使用合成的内省数据以比约束系统提示或激活引导等替代方法更有效和可控的方式塑造助手角色。具体而言，我们使用11个示例角色（如幽默、深切关怀甚至恶意）对三种流行的开放权重模型进行微调。为了追踪我们方法的效果，我们引入了一种分析显现偏好的方法，揭示角色在整体上的明显变化。我们发现这些变化对对抗性提示的鲁棒性高于上述两种替代方法，同时也导致生成内容更加连贯和真实。最后，我们证明这种微调对常见基准测量的通用能力几乎没有影响。我们描述并开源了我们完整的后训练方法，其实现可在此HTTPS网址找到。",
      "paper_summary": {
        "summary": "Researchers introduce the first open-source methodology for \"character training\" AI assistants, leveraging Constitutional AI principles and a novel three-stage pipeline. The method cultivates deep, robust, and coherent personas in open-weights Large Language Models while preserving general capabilities, assessed via a new \"revealed preferences\" evaluation approach.",
        "originalProblem": [
          "The process of \"character training\" AI assistant personas in LLMs, crucial for user experience and alignment, remains largely proprietary and unstudied in academic literature.",
          "Existing academic approaches for persona shaping, such as human-centric psychometrics or inference-time prompting, are superficial, brittle, and fail to embed robust, coherent character traits.",
          "Reliably evaluating deep, holistic changes in AI personas is challenging, as traditional self-report psychometrics often show weak correlations with perceived behavior."
        ],
        "solution": [
          "A three-stage sequential pipeline is introduced for open character training: hand-written \"constitutions\" defining desired traits, DPO-based distillation from a teacher model (GLM 4.5 AIR), and fine-tuning with synthetic introspective data (self-reflection and self-interaction).",
          "The full methodology, code, model checkpoints, and training data are openly released, enabling the shaping of 11 diverse personas (e.g., Sarcastic, Humorous, Misaligned) into popular open-weights LLMs (Qwen 2.5, Llama 3.1, Gemma 3).",
          "A novel \"revealed preferences\" evaluation method is developed where an LLM-as-a-Judge assesses which of two single-word trait descriptors a model implicitly embodied, overcoming limitations of self-reports."
        ],
        "keyInsights": [
          "A multi-stage training approach, particularly incorporating synthetic introspective data, is essential for embedding deep, robust, and coherent AI personas that are more resilient to adversarial attacks.",
          "Constitutional AI principles can be effectively adapted and open-sourced, democratizing advanced persona shaping techniques previously confined to proprietary frontier AI labs.",
          "The \"revealed preferences\" evaluation method provides a more objective and fine-grained way to measure persona changes, demonstrating intuitive control over desired and opposing traits."
        ],
        "results": [
          "Character training consistently boosts desired traits and suppresses opposing ones, leading to intuitive and fine-grained control over persona; trait preferences across different base models significantly converge (average Spearman correlation from 0.44 to 0.87).",
          "Models trained with the full pipeline demonstrate superior robustness to adversarial prompting and multi-turn prefill attacks compared to baselines using system prompts or activation steering, with introspective data particularly enhancing this robustness.",
          "The character training method maintains models' general capabilities across five standard LLM benchmarks, with degradation only observed when explicitly encouraged by \"misaligned\" personas."
        ]
      },
      "image_url": "image/2511.01689v1.png",
      "universal_paper_id": "2511.01689",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 55,
          "last_7_days": 55
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-11-03T15:53:47.000Z",
      "publication_date": "2025-11-03T15:53:47.000Z",
      "updated_at": "2025-11-04T08:47:17.156Z",
      "topics": [
        "adversarial-robustness",
        "agents",
        "Computer Science",
        "conversational-ai",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "fine-tuning",
        "human-ai-interaction",
        "model-interpretation",
        "synthetic-data",
        "text-generation",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 8,
      "github_url": "https://github.com/maiush/OpenCharacterTraining",
      "distance": 1
    },
    {
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们介绍Emu3.5，这是一个大规模多模态世界模型，能够原生地预测视觉和语言的下一个状态。Emu3.5通过一个统一的下一个标记预测目标进行了端到端的预训练，数据集包含超过10万亿个标记，主要来自互联网视频的连续帧和文本记录。该模型自然接受交错的视觉-语言输入，并生成交错的视觉-语言输出。Emu3.5随后通过大规模强化学习进行了后续训练，以增强多模态推理和生成。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐标记解码转换为双向并行预测，约提高每张图像的推理速度20倍，而不牺牲性能。Emu3.5展现出强大的原生多模态能力，包括长时间范围的视觉-语言生成、任意到图像（X2I）生成和复杂的富文本图像生成。它还表现出可泛化的世界建模能力，使得在多种场景和任务中能够进行时空一致的世界探索和开放世界的具身操作。作为比较，Emu3.5在图像生成和编辑任务中的表现与Gemini 2.5 Flash Image（Nano Banana）相当，并在一系列交错生成任务中展现出更优的结果。我们在此网址开源Emu3.5，以支持社区研究。",
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
          "all": 764,
          "last_7_days": 764
        },
        "public_total_votes": 80
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
      "id": "019a4d54-272a-7d61-9712-1d9963161888",
      "paper_group_id": "019a4d54-272a-7d61-9712-1d9963161888",
      "title": "Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process",
      "abstract": "视觉-语言-动作（VLA）模型旨在理解自然语言指令和视觉观察，并作为具身代理执行相应的动作。最近的研究将未来图像整合到理解-行动循环中，产生了统一的VLA，能够共同理解、生成和行动——读取文本和图像，生成未来的图像和动作。然而，这些模型要么依赖外部专家进行模态统一，要么将图像生成和动作预测视为独立的过程，从而限制了这两项任务之间直接协同的好处。我们的核心理念是通过同步去噪过程共同优化生成与动作，其中迭代精炼使得动作在持续且充分的视觉引导下，从初始化中演变。我们根据所提出的统一扩散VLA和联合离散去噪扩散过程（JD3P）来扎根于这一理念，该过程是一个联合扩散过程，将多种模态整合到一个单一的去噪轨迹中，以作为理解、生成和行动内在协同的关键机制。我们的模型和理论建立在一个统一的所有模态的标记空间和混合注意机制之上。我们进一步提出了一个两阶段的训练流程和若干推理时间技术，以优化性能和效率。我们的方法在CALVIN、LIBERO和SimplerEnv等基准上达到了最先进的性能，其推理速度比自回归方法快4倍，并通过深入分析和现实世界评估证明了其有效性。我们的项目页面可在该Https URL上找到。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 56,
          "last_7_days": 56
        },
        "public_total_votes": 11
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
      "id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "paper_group_id": "019a47ab-b12a-7771-a0dd-e19b18260bfe",
      "title": "The Denario project: Deep knowledge AI agents for scientific discovery",
      "abstract": "我们介绍了Denario，一个旨在作为科学研究助手的AI多智能体系统。Denario可以执行多种不同的任务，比如生成创意、查阅文献、制定研究计划、编写和执行代码、制作图表，以及撰写和审阅科学论文。该系统采用模块化架构，可以处理特定任务，例如生成创意，或使用Cmbagent作为深度研究后端进行端到端的科学分析。在这项工作中，我们详细描述了Denario及其模块，并通过展示其在天体物理学、生物学、生物物理学、生物医学信息学、化学、材料科学、数学物理学、医学、神经科学和行星科学等多个科学领域生成的多篇AI论文来说明其能力。Denario还擅长于结合不同学科的想法，我们通过展示一篇将量子物理和机器学习方法应用于天体物理数据的论文来说明这一点。我们报告了领域专家对这些论文所进行的评估，专家提供了数值评分以及类似审稿的反馈意见。然后，我们强调了当前系统的优点、缺点和局限性。最后，我们讨论了AI驱动研究的伦理影响，并反思这种技术与科学哲学的关系。我们在这个HTTPS链接上公开发布了代码。Denario的演示也可以在这个HTTPS链接上直接运行，完整应用程序将部署在云端。",
      "paper_summary": null,
      "image_url": "image/2510.26887v1.png",
      "universal_paper_id": "2510.26887",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 197,
          "last_7_days": 197
        },
        "public_total_votes": 22
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
      "id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "paper_group_id": "019a380e-b464-7026-a226-547dfe0bbcfa",
      "title": "Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning",
      "abstract": "大型语言模型（LLMs）常常在需要多步骤推理的问题上表现不佳。对于小规模的开源模型，基于可验证奖励的强化学习（RLVR）在正确解决方案很少被采样的情况下，即使经过多次尝试也会失败，而监督微调（SFT）则倾向于通过严格的逐 token 模仿来过拟合长示范。为了解决这个问题，我们提出了监督强化学习（SRL），这是一个将问题解决重新表述为生成一系列逻辑“动作”的框架。SRL 训练模型在执行每个动作之前生成内部推理独白。它根据模型的动作与从 SFT 数据集中提取的专家动作之间的相似性，以逐步的方式提供更平滑的奖励。这种监督即使在所有 rollout 均不正确的情况下，也提供了更丰富的学习信号，同时鼓励根据专家示范灵活推理。因此，SRL 使小模型能够学习以前无法通过 SFT 或 RLVR 学习的复杂问题。此外，在使用 RLVR 进行精细化之前，先使用 SRL 初始化训练能获得最强的整体性能。在推理基准之外，SRL 在自主软件工程任务中也表现出有效的迁移能力，确立了其作为面向推理的 LLMs 训练框架的稳健性和多样性。",
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
        "total_votes": 16,
        "visits_count": {
          "all": 781,
          "last_7_days": 781
        },
        "public_total_votes": 75
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
      "id": "019a4faa-3a91-7dfb-a787-186de6f8b564",
      "paper_group_id": "019a4faa-3a91-7dfb-a787-186de6f8b564",
      "title": "MotionStream: Real-Time Video Generation with Interactive Motion Controls",
      "abstract": "当前运动条件视频生成方法存在过高的延迟（每个视频数分钟）和非因果处理，无法实现实时交互。我们提出了MotionStream，使得在单个GPU上能够实现亚秒级延迟，达到每秒最多29帧的流生成。我们的方法首先通过运动控制增强了文本到视频模型，该模型生成遵循全局文本提示和局部运动指导的高质量视频，但不进行即时推断。因此，我们通过分布匹配蒸馏的自我强迫，将这种双向教师蒸馏为因果学生，从而实现实时流推断。在生成长时间（可能是无限）视频时，出现了几个关键挑战：（1）弥合在有限长度上训练与外推到无限视域之间的领域差距，（2）通过防止错误积累来保持高质量，以及（3）维持快速推断，而不因上下文窗口的增加而导致计算成本增长。我们方法的关键是引入精心设计的滑动窗口因果注意力，并结合注意力汇聚。通过在训练期间结合具有注意力汇聚的自回滚和KV缓存滚动，我们能够在固定上下文窗口下适当模拟推断时的外推，使得可以以恒定速度生成任意长度的视频。我们的模型在运动跟随和视频质量方面达到了最先进的结果，同时速度快了两个数量级，独特地支持无限长度的流媒体生成。通过MotionStream，用户可以绘制轨迹、控制摄像机或转移运动，并实时看到结果展开，提供真正的互动体验。",
      "paper_summary": {
        "summary": "MotionStream, developed by researchers from Adobe Research, Carnegie Mellon University, and Seoul National University, introduces a framework for real-time, interactive, and infinite-length video generation with motion controls. It achieves up to 29.5 FPS with 0.39s latency on a single H100 GPU, enabling users to continuously guide photorealistic video generation with immediate results.",
        "originalProblem": [
          "Existing motion-controlled video generation methods are offline and non-causal, requiring full motion input upfront and resulting in prohibitive latency (e.g., minutes for a 5-second clip).",
          "Prior autoregressive video models struggle with maintaining long-term consistency and generating high-quality content over extended durations, often exhibiting issues like color drifts.",
          "Interactive video world models often demand substantial computational resources for inference or are limited to closed-domain or synthetic environments."
        ],
        "solution": [
          "A two-stage pipeline is used: first, a bidirectional motion-controlled teacher model is trained with a lightweight sinusoidal positional encoding for track representation and joint text-motion guidance.",
          "This high-quality teacher is then distilled into a fast, causal student model for streaming generation, employing autoregressive roll-out with rolling Key-Value (KV) caches and attention sinks.",
          "An optimized 'Tiny VAE' decoder is integrated to significantly reduce video decoding time, further boosting overall frame rates and reducing latency for real-time performance."
        ],
        "keyInsights": [
          "A lightweight track representation using sinusoidal positional embeddings and a learnable track head proves two orders of magnitude faster (24.8ms) than VAE-based RGB encoding while maintaining motion accuracy.",
          "The 'attention sink' mechanism, adapted from StreamingLLMs, is critical for preventing quality degradation and drift during long-video extrapolation, with a minimal single-chunk sink proving most effective.",
          "Explicitly simulating autoregressive inference dynamics, including self-rollout with rolling KV caches and attention sinks, during student training is crucial for a perfect train-test match and stable long-term generation."
        ],
        "results": [
          "Achieves up to 29.5 FPS with 0.39s latency on a single H100 GPU for 480P video, outperforming prior methods in speed by two orders of magnitude while demonstrating better motion accuracy (e.g., EPE of 7.80 on DAVIS).",
          "The track-conditioned causal model significantly outperforms specialized 3D novel view synthesis baselines in visual fidelity (PSNR, SSIM, LPIPS) on the LLFF dataset at 16.7 FPS.",
          "Effectively supports real-time interactive applications, including online motion transfer, user drag operations, and 3D camera control, allowing continuous guidance and immediate visual feedback."
        ]
      },
      "image_url": "image/2511.01266v1.png",
      "universal_paper_id": "2511.01266",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 34,
          "last_7_days": 34
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-11-03T06:37:53.000Z",
      "publication_date": "2025-11-03T06:37:53.000Z",
      "updated_at": "2025-11-04T16:19:07.537Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.LG",
        "efficient-transformers",
        "generative-models",
        "human-ai-interaction",
        "inference-optimization",
        "knowledge-distillation",
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
      "id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "paper_group_id": "019a383a-e3e1-778f-9603-1e1f3b783290",
      "title": "The Era of Agentic Organization: Learning to Organize with Language Models",
      "abstract": "我们设想一个新的人工智能时代，称为代理组织，在这个时代，代理通过协作和并行工作解决复杂问题，从而实现超越个体智慧的结果。为了实现这一愿景，我们引入了异步思维（AsyncThink），作为一种新的推理范式，与大型语言模型结合，组织内部思维过程为可并行执行的结构。具体来说，我们提出了一种思维协议，其中组织者动态地将子查询分配给工作者，合并中间知识，并生成连贯的解决方案。更重要的是，这种思维结构可以通过强化学习进一步优化。实验表明，与并行思维相比，AsyncThink 的推理延迟降低了 28%，同时提高了数学推理的准确性。此外，AsyncThink 还将其学习到的异步思维能力进行了泛化，能够有效应对未见过的任务，而无须额外的训练。",
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
        "total_votes": 8,
        "visits_count": {
          "all": 442,
          "last_7_days": 442
        },
        "public_total_votes": 46
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
      "abstract": "现代大型语言模型（LLM）主要通过显式文本生成的方式“思考”，例如链式推理（CoT），这将推理推迟到训练后，并未充分利用预训练数据。我们提出并开源了Ouro，命名源自递归的乌鲁波罗斯，它是一系列预训练的循环语言模型（LoopLM），通过（i）在潜在空间中进行迭代计算，（ii）用于学习深度分配的熵正则化目标，以及（iii）扩展到7.7万亿个标记，将推理构建到预训练阶段。Ouro 1.4B和2.6B模型在广泛的基准测试中表现优越，达到了高达12B的最新技术水平大型语言模型的结果。通过控制实验，我们表明这一优势并非来源于知识容量的提升，而是优越的知识操控能力。我们还表明，LoopLM产生的推理痕迹与最终输出的对齐程度高于显式的CoT。我们希望我们的结果展示了LoopLM在推理时代作为一种新兴扩展方向的潜力。我们的模型可以在：这个HTTP网址中找到。",
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
        "total_votes": 27,
        "visits_count": {
          "all": 1142,
          "last_7_days": 1142
        },
        "public_total_votes": 106
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
      "id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "paper_group_id": "019a47e0-5c4d-73b6-8c02-d5000852a3b2",
      "title": "ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning",
      "abstract": "多模态推理需要语言和视觉之间的迭代协调，但尚不清楚什么构成了有意义的交织思维链。我们认为，文本和图像思维应作为互补而非同构的模态，相互促进推理。在这一原则指导下，我们构建了ThinkMorph，这是一个统一模型，经过约24,000个高质量交织推理痕迹的精细调优，涵盖了不同视觉参与度的任务。ThinkMorph学习生成逐步的文本-图像推理步骤，具体操作视觉内容的同时保持连贯的语言逻辑。它在以视觉为中心的基准测试上取得了显著的提升（相较于基础模型平均提高34.7%），并能够推广到领域外的任务，达到或超过更大且专有的视觉语言模型（VLM）。除了性能之外，ThinkMorph还展现了新兴的多模态智能，包括未见的视觉操作技能、在推理模式之间自适应切换的能力，以及通过多样化多模态思维实现更好的测试时间扩展。这些发现为表征统一模型在多模态推理中的新兴能力提出了有前景的方向。",
      "paper_summary": null,
      "image_url": "image/2510.27492v2.png",
      "universal_paper_id": "2510.27492",
      "metrics": {
        "total_votes": 4,
        "visits_count": {
          "all": 210,
          "last_7_days": 210
        },
        "public_total_votes": 26
      },
      "first_publication_date": "2025-10-30T17:51:38.000Z",
      "publication_date": "2025-11-04T13:29:38.000Z",
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
      "id": "019a4d56-ae23-7d39-8e43-a906e66140b1",
      "paper_group_id": "019a4d56-ae23-7d39-8e43-a906e66140b1",
      "title": "PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model",
      "abstract": "视觉-语言-动作模型（VLA）正作为学习可推广的视觉运动控制策略的强大工具而兴起。然而，目前的 VLA 在两方面仍然有限：（i）它们在像素级场景理解上存在困难，(ii) 它们过于依赖文本提示，这降低了在现实世界环境中的灵活性。为了解决这些挑战，我们推出了 PixelVLA，这是首个旨在支持像素级推理和文本及视觉输入多模态提示的 VLA 模型。我们的方法基于一个新的视觉运动指令调优框架，将多尺度像素感知编码器与视觉提示编码器相结合。为了有效训练 PixelVLA，我们进一步提出了一个两阶段自动注释管道，生成 Pixel-160K，这是一个大规模数据集，包含来自现有机器人数据的像素级注释。在三个标准 VLA 基准和两个 VLA 模型变体上的实验证明，PixelVLA 相较于 OpenVLA 提高了 10.1%-17.8% 的操作成功率，同时只需 1.5% 的预训练成本。这些结果表明，PixelVLA 可以集成到现有 VLA 中，从而在复杂环境中实现更准确、高效和多功能的机器人控制。数据集和代码将作为开源发布。",
      "paper_summary": null,
      "image_url": "image/2511.01571v1.png",
      "universal_paper_id": "2511.01571",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 31,
          "last_7_days": 31
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-11-03T13:39:37.000Z",
      "publication_date": "2025-11-03T13:39:37.000Z",
      "updated_at": "2025-11-04T05:28:37.667Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.RO",
        "data-curation",
        "fine-tuning",
        "image-segmentation",
        "imitation-learning",
        "multi-modal-learning",
        "robotic-control",
        "robotics-perception",
        "vision-language-models"
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
      "abstract": "空间理解仍然是大型视觉语言模型（LVLMs）的一个弱点。现有的监督微调（SFT）和最近的可验证奖励强化学习（RLVR）流程依赖于昂贵的监督、专业工具或受限环境，这限制了规模。我们推出了Spatial-SSRL，这是一种自监督的强化学习范例，直接从普通的RGB或RGB-D图像中获取可验证信号。Spatial-SSRL自动 formulates 了五个前置任务，以捕捉二维和三维空间结构：打乱的补丁重排序、翻转补丁识别、裁剪补丁修复、区域深度排序以及相对三维位置预测。这些任务提供了易于验证的真实答案，且不需要人工或LVLM的标注。在我们的任务上训练显著提高了空间推理能力，同时保留了通用的视觉能力。在图像和视频设置的七个空间理解基准上，Spatial-SSRL相较于Qwen2.5-VL基线平均提升了4.63%（3B）和3.89%（7B）的准确率。我们的结果表明，简单的内在监督使得大型RLVR成为可能，并为LVLMs提供了一条实现更强空间智能的实用路径。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 140,
          "last_7_days": 140
        },
        "public_total_votes": 22
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
      "abstract": "强化学习（RL）对大型语言模型（LLM）进行微调时常常面临不稳定性问题，这主要是因为训练和推理策略之间的数值不匹配。虽然之前的工作尝试通过算法修正或工程对齐来缓解这一问题，但我们展示了其根本原因在于浮点精度本身。尽管广泛采用的BF16具有较大的动态范围，但它会引入较大的舍入误差，从而破坏训练和推理之间的一致性。在本研究中，我们证明简单恢复使用\\textbf{FP16}可以有效消除这种不匹配。这一变化简单，现代框架完全支持，只需修改几行代码，并且不需要对模型架构或学习算法进行任何修改。我们的结果表明，均匀使用FP16可以实现更稳定的优化、更快的收敛速度以及在各种任务、算法和框架下更强的性能。我们希望这些发现能够激励人们更广泛地重新考虑RL微调中的精度权衡问题。",
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
        "total_votes": 28,
        "visits_count": {
          "all": 741,
          "last_7_days": 741
        },
        "public_total_votes": 92
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
      "id": "019a52d6-c209-7788-8f67-88c7d50e50b1",
      "paper_group_id": "019a52d6-c209-7788-8f67-88c7d50e50b1",
      "title": "VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation",
      "abstract": "代码已成为智能体时代推理和行动的精确可执行媒介。然而，迄今为止，发展主要集中在语言驱动的任务上，如程序合成和调试，而视觉驱动的编码却未得到充分探讨。受到人类如何对草图进行推理的启发，我们提倡使用SVG代码作为一种紧凑、可解释和可执行的视觉表达方式。我们引入了VCode，一个将多模态理解重构为代码生成的基准：给定一张图像，模型必须生成保持符号意义的SVG，以便于后续推理。VCode涵盖三个领域——一般常识（MM-Vet）、专业学科（MMMU）和视觉驱动的感知（CV-Bench）。为了评估符号的保真度，我们提出了CodeVQA，一种新的评估协议，其中策略模型针对渲染的SVG回答问题；正确答案表明符号得到了忠实保留。在实证中，前沿的视觉语言模型在生成忠实的SVG方面仍存在困难，揭示了语言驱动编码与视觉驱动编码之间的持续差距。为缩小这一差距，我们引入了VCoder，一个在两个方向上增强视觉语言模型的智能框架：（i）通过修订进行思考，迭代分析差异并完善SVG代码；以及（ii）使用视觉工具进行行动，其中探测器和解析器提供结构化线索，如物体、形状和文本，超出了模型的内在能力。在各项基准测试中，具有强大推理能力的前沿视觉语言模型整体得分良好，但在专业知识和3D推理方面仍然有限。VCoder在顶级的Claude-4-Opus上提供了12.3分的整体提升。人类研究表明，无论是人类还是视觉语言模型在渲染的SVG上表现都较差，他们的一致性揭示了符号视觉表现的潜力。该基准和代码可在此HTTPS链接获取。",
      "paper_summary": {
        "summary": "The VCode project introduces a multimodal coding benchmark that requires Vision-Language Models to translate natural images into Scalable Vector Graphics (SVG) code, providing a symbolic and executable visual representation. The proposed VCoder framework, which employs iterative revision and external visual tools, improves state-of-the-art VLMs by 12.3 CodeVQA points on this challenging task.",
        "originalProblem": [
          "Existing coding benchmarks primarily evaluate language models on textual code without visual input.",
          "Multimodal understanding benchmarks typically rely on natural language responses rather than generating symbolic visual representations in code.",
          "Current pixel-based image representations offer limited symbolic abstraction, which is crucial for human-like visual reasoning and agentic action."
        ],
        "solution": [
          "Introduces VCode, a benchmark that redefines multimodal understanding as a code generation task, requiring models to produce SVG from natural images that preserves symbolic meaning.",
          "Proposes CodeVQA, a novel evaluation protocol that assesses the symbolic fidelity of generated SVG by having a policy model answer questions about the original image using only the rendered SVG.",
          "Develops VCoder, an agentic framework that enhances VLMs through 'Thinking with Revision' (iterative refinement with rendering feedback) and 'Acting with Visual Tools' (integration of object detection, segmentation, and OCR)."
        ],
        "keyInsights": [
          "A substantial gap persists between current state-of-the-art VLMs' language-centric and visual-centric coding capabilities, even for frontier models like GPT-5.",
          "Models with stronger general reasoning consistently achieve better visual coding scores, and a positive correlation exists between semantic similarity and symbolic fidelity (CodeVQA performance).",
          "Long-context code generation is a critical bottleneck, as models producing longer, more detailed SVG sequences tend to perform better."
        ],
        "results": [
          "Frontier VLMs like GPT-5 achieve a CodeVQA score of 46.8, notably below the 61.7 score obtained by reasoning directly over raw images.",
          "The VCoder framework, built on Claude-4-Opus, delivers an overall gain of 12.3 CodeVQA points, improving its score from 41.7 to 54.0 across diverse domains.",
          "The ensemble of visual tools (category, location, shape, text) provides a significant 16.6-point improvement in CodeVQA over the base VLM, demonstrating the power of structured visual cues."
        ]
      },
      "image_url": "image/2511.02778v1.png",
      "universal_paper_id": "2511.02778",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 22,
          "last_7_days": 22
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-11-04T18:00:18.000Z",
      "publication_date": "2025-11-04T18:00:18.000Z",
      "updated_at": "2025-11-05T07:06:37.449Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.CL",
        "cs.CV",
        "image-generation",
        "multi-modal-learning",
        "reasoning",
        "representation-learning",
        "tool-use",
        "vision-language-models",
        "visual-qa"
      ],
      "organization_info": [
        {
          "name": "University of Oxford",
          "image": "images/organizations/oxford.jpg"
        },
        {
          "name": "University of Science and Technology of China",
          "image": "images/organizations/university-of-science-and-technology-of-china.svg+xml"
        },
        {
          "name": "Central South University",
          "image": null
        },
        {
          "name": "Microsoft",
          "image": "images/organizations/microsoft.png"
        }
      ],
      "author_info": [],
      "github_stars": 29,
      "github_url": "https://github.com/CSU-JPG/VCode",
      "distance": 1
    },
    {
      "id": "019a503d-c90c-77af-89dd-6eb6c4e45419",
      "paper_group_id": "019a503d-c90c-77af-89dd-6eb6c4e45419",
      "title": "ROVER: Benchmarking Reciprocal Cross-Modal Reasoning for Omnimodal Generation",
      "abstract": "统一多模态模型（UMMs）作为一种强大的范式，已迅速发展成为无缝统一文本与图像理解和生成的工具。然而，现有的评估方法将这些能力孤立看待，使得涉及多模态输入和输出的任务主要通过单模态推理进行评分，即文本基准强调以语言为基础的推理，而视觉基准则强调以像素体现的推理结果。我们推出了ROVER，以应对测试互惠交叉模态推理这一迫切需求，即一种利用一种模态来指导、验证或完善另一种模态的输出的能力，这是统一多模态智能愿景的核心。ROVER是一个经过人工标注的基准，专门针对互惠交叉模态推理，其中包含1312个基于1876张图像的任务，涵盖了两种互补的设置。用于视觉生成的口头增强推理评估模型是否可以使用口头提示和推理链来指导忠实的图像合成。用于口头生成的视觉增强推理评估模型是否可以生成中间可视化，以加强其自身的推理过程以解答问题。在17个统一模型上的实验揭示了两个关键发现：（i）交叉模态推理决定视觉生成的质量，交错模型显著优于非交错模型；值得注意的是，简单结合强大的单模态模型并未达到可比的推理效果。（ii）模型在物理推理和象征推理之间存在解离：它们成功地对感知概念进行字面解释，但在构建象征性任务的视觉抽象时失败，错误的推理影响了性能。这些结果突显了互惠交叉模态推理作为实现真正全模态生成的关键前沿。",
      "paper_summary": {
        "summary": "The ROVER benchmark evaluates the reciprocal cross-modal reasoning abilities of unified multimodal models across visual and verbal generation tasks. It identified a clear dissociation in models' capabilities, showing proficiency in physical reasoning but significant struggles with abstract and symbolic visual reasoning.",
        "originalProblem": [
          "Unified Multimodal Models (UMMs) lacked benchmarks to assess their reciprocal cross-modal reasoning, where one modality guides or refines the other.",
          "Prior evaluations primarily assessed unimodal abilities in isolation, failing to capture the synergistic intelligence needed for seamless understanding and generation across text and images.",
          "Existing methods often overlooked the logical coherence of the reasoning process, focusing instead on output-level metrics."
        ],
        "solution": [
          "Introduced ROVER, a human-annotated benchmark with 1,312 tasks across two settings: verbally-augmented reasoning for visual generation (ROVER-IG) and visually-augmented reasoning for verbal generation (ROVER-TG).",
          "Designed a principled task taxonomy covering various conceptual domains and reasoning subtasks, ensuring visual outputs actively aid reasoning.",
          "Developed a multi-dimensional evaluation protocol utilizing GPT-4.1 as an automated VLM-as-judge, assessing reasoning process quality, cross-modal alignment, and content consistency."
        ],
        "keyInsights": [
          "Cross-modal reasoning capabilities directly determine the quality of visual generation, with better reasoning leading to superior visual outputs.",
          "Current UMMs exhibit a notable dissociation: they perform well on physical world and visual perception tasks but struggle fundamentally with abstract and symbolic visual reasoning (e.g., in logic and mathematics).",
          "Interleaved image-text generation significantly enhances visual generation performance, suggesting benefits from integrating reasoning steps across modalities."
        ],
        "results": [
          "Closed-source models demonstrated ~38% higher verbal reasoning and ~31% better alignment scores, resulting in ~39% improved visual generation compared to open-source models.",
          "Unified models showed limited capacity for meaningful visual reasoning steps on ROVER-TG, achieving only 38.8% average Interleaved Reasoning Quality.",
          "Visual augmentation improved accuracy on physical world and visual perception tasks (by 3.5% to 3.8%) but degraded performance on logic and math tasks (by 1.4%) when visual reasoning was of low quality."
        ]
      },
      "image_url": "image/2511.01163v1.png",
      "universal_paper_id": "2511.01163",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 22,
          "last_7_days": 22
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-11-03T02:27:46.000Z",
      "publication_date": "2025-11-03T02:27:46.000Z",
      "updated_at": "2025-11-04T19:00:17.804Z",
      "topics": [
        "chain-of-thought",
        "Computer Science",
        "cs.CV",
        "generative-models",
        "image-generation",
        "multi-modal-learning",
        "reasoning",
        "text-generation",
        "vision-language-models",
        "visual-qa"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/cheryyunl/ROVER",
      "distance": 1
    },
    {
      "id": "019a530b-216e-74d8-9f83-552c9844e288",
      "paper_group_id": "019a530b-216e-74d8-9f83-552c9844e288",
      "title": "TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System",
      "abstract": "大规模数据推动了机器人技术的突破，从语言模型到双手操作中的视觉-语言-行动模型。然而，人形机器人在数据收集框架方面缺乏同样有效的方案。现有的人形遥操系统要么使用解耦控制，要么依赖昂贵的动作捕捉设备。我们介绍了TWIST2，一个便携式的无动作捕捉人形遥操与数据收集系统，它在推进可扩展性的同时保持了全身的完全控制。我们的系统利用PICO4U VR实时获取全身的人类动作，并配备一个自定义的2自由度机器人颈部（成本约为250美元）用于自我中心的视觉，实现了整体人类与人形的控制。我们展示了长时间的灵巧和移动人形技能，并且能够在15分钟内收集100个示范，成功率几乎达到100%。在此基础上，我们提出了一个自给自足的分层视觉运动策略框架，可以基于自我中心视觉自主控制整个模拟人类的身体。我们的视觉运动策略成功演示了全身灵巧操作和动态踢球任务。整个系统是完全可复制的，并已在此网址开源。我们收集的数据集也已在此网址开源。",
      "paper_summary": {
        "summary": "TWIST2 provides a scalable and portable system for collecting holistic humanoid data, leveraging affordable VR and a custom robot neck for whole-body teleoperation with active egocentric vision. This approach allows efficient generation of diverse human demonstrations, exemplified by collecting 100 bimanual pick & place demonstrations in 18.5 minutes, and enables training real-world autonomous policies.",
        "originalProblem": [
          "Humanoid robotics research faces a scarcity of high-quality, large-scale demonstration data, limiting the application of data-driven learning methods.",
          "Existing full whole-body teleoperation systems for humanoids depend on expensive, non-portable motion capture setups, restricting data collection to specialized lab environments.",
          "Developing general visuomotor policies for humanoids is challenging due to the need for coordinated whole-body control and integrated egocentric vision for complex tasks."
        ],
        "solution": [
          "Developed TWIST2, a portable, low-cost human data source that replaces traditional MoCap by using a PICO 4U VR headset and ankle trackers for real-time full human body pose estimation.",
          "Designed and integrated a custom, low-cost 2-DoF neck module for the Unitree G1 humanoid robot, equipped with a ZED Mini stereo camera to provide active egocentric stereo vision.",
          "Implemented a two-level hierarchical control framework comprising a low-level reinforcement learning-trained motion tracker and a high-level command generator, either from human teleoperation or a visuomotor policy."
        ],
        "keyInsights": [
          "Combining consumer-grade VR hardware with a custom active vision module enables highly effective, portable, and scalable whole-body teleoperation for humanoid robots.",
          "Egocentric active perception and smooth whole-body tracking are crucial for allowing humanoids to execute natural, long-horizon, and mobile manipulation tasks efficiently.",
          "Hierarchical visuomotor policies, trained on holistically collected data, can achieve vision-based autonomous control for diverse and complex whole-body humanoid tasks in real-world scenarios."
        ],
        "results": [
          "TWIST2 successfully demonstrated teleoperation of complex, long-horizon whole-body dexterous tasks, such as folding towels and mobile object transport through doors.",
          "The system achieved high data collection efficiency, with an expert teleoperator collecting approximately 100 bimanual pick & place demonstrations in 18.5 minutes (11 seconds/episode) with a 100% success rate.",
          "Autonomous policies, trained on data collected via TWIST2, successfully performed real-world tasks like whole-body dexterous pick & place and kicking a T-shaped box to a target, demonstrating robust autonomous capabilities."
        ]
      },
      "image_url": "image/2511.02832v1.png",
      "universal_paper_id": "2511.02832",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 20,
          "last_7_days": 20
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-11-04T18:58:35.000Z",
      "publication_date": "2025-11-04T18:58:35.000Z",
      "updated_at": "2025-11-05T08:03:49.742Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.CV",
        "cs.LG",
        "cs.RO",
        "data-curation",
        "deep-reinforcement-learning",
        "human-ai-interaction",
        "imitation-learning",
        "multi-modal-learning",
        "robotic-control",
        "robotics-perception"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a51a1-893a-73b2-af52-16b3948d7168",
      "paper_group_id": "019a51a1-893a-73b2-af52-16b3948d7168",
      "title": "Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems",
      "abstract": "最近在大型语言模型多智能体系统（LLM Multi-Agent Systems）方面的进展，使得对子智能体的可扩展协调成为可能，每个智能体可以协调数百或数千个工具或模型上下文协议（MCP）服务器。然而，现有的检索方法通常在路由之前将查询与粗略的智能体级描述进行匹配，这模糊了细粒度工具功能，并经常导致次优的智能体选择。我们引入了工具到智能体检索（Tool-to-Agent Retrieval），这是一个统一框架，它将工具及其父智能体嵌入到共享的向量空间中，并通过元数据关系将它们连接起来。通过明确表示工具的能力并遍历元数据到达智能体级别，工具到智能体检索实现了细粒度的工具级或智能体级检索，确保智能体及其底层工具或MCP服务器得到平等的表示，而不会由于将许多工具归为一类而导致上下文稀释。我们在八个嵌入模型上评估工具到智能体检索的方法，结果显示，在LiveMCPBench基准上，相较于以前的最先进智能体检索器，我们的方法在Recall@5上 consistently 提高了19.4%，在nDCG@5上提高了17.7%。",
      "paper_summary": null,
      "image_url": "image/2511.01854v1.png",
      "universal_paper_id": "2511.01854",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 18,
          "last_7_days": 18
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-11-03T18:58:28.000Z",
      "publication_date": "2025-11-03T18:58:28.000Z",
      "updated_at": "2025-11-05T01:28:52.282Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.CL",
        "embedding-methods",
        "ml-systems",
        "optimization-methods",
        "representation-learning",
        "tool-use"
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
      "abstract": "在序列建模中，原子事实的参数化记忆主要被抽象为实体之间共现的蛮力查找。我们将这种关联视角与记忆存储的几何视角进行对比。我们首先隔离出一个干净且可分析的 Transformer 推理实例，这与记忆仅仅是训练期间指定的局部共现的存储是不兼容的。相反，模型必须以某种方式合成其自身的原子事实几何，编码所有实体之间的全局关系，包括不共现的实体。这反过来将一个涉及 $\\ell$ 次组合的困难推理任务简化为一个容易学习的一步几何任务。\n\n从这一现象中，我们提取出神经嵌入几何中难以解释的基本方面。我们认为，这种几何的兴起，尽管仅优化局部关联，却不能简单地归因于典型的架构或优化压力。出人意料的是，即使在其表达并不比蛮力查找关联更简洁的情况下，优雅的几何也被学习到了。\n\n然后，通过分析与 Node2Vec 的联系，我们展示了几何是如何源于一种谱偏差——与现有理论相对立——实际上在缺乏多种压力的情况下自然而然地产生。这一分析还为从业者指明了如何使 Transformer 记忆更具几何性质的明显空间。我们希望参数化记忆的几何视角能鼓励重新审视指导研究者在知识获取、容量、发现和遗忘等领域的默认直觉。",
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
          "all": 293,
          "last_7_days": 293
        },
        "public_total_votes": 42
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
      "id": "019a4d61-3d65-7239-87e3-f41e9e98d86f",
      "paper_group_id": "019a4d61-3d65-7239-87e3-f41e9e98d86f",
      "title": "Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail",
      "abstract": "通过模仿学习训练的端到端架构在自动驾驶领域取得了进展，扩大了模型规模和数据，但在安全关键的长尾场景中，性能仍然脆弱，因为监督稀缺且因果理解有限。为此，我们引入了Alpamayo-R1（AR1），这是一个视觉-语言-动作模型（VLA），它结合了因果链推理与轨迹规划，以增强复杂驾驶场景中的决策能力。我们的方法具有三个关键创新点：(1) 因果链（CoC）数据集，通过混合自动标注和人机协作的流程构建，生成与驾驶行为对齐的，基于决策的因果推理轨迹；(2) 模块化的VLA架构，将为物理人工智能应用预训练的视觉-语言模型Cosmos-Reason与基于扩散的轨迹解码器结合，实时生成动态可行的规划；(3) 多阶段训练策略，采用监督微调以引导推理，并通过强化学习（RL）优化推理质量，借助大型推理模型反馈并强化推理-动作一致性。评估显示，与仅使用轨迹的基线相比，AR1在具有挑战性的案例中实现了高达12%的规划准确率提升，同时在闭环模拟中，越界率降低了35%，近距离碰撞率降低了25%。RL后训练通过大型推理模型评估，推理质量提升了45%，推理-动作一致性提升了37%。模型参数从0.5B扩展到7B显示出一致的改进。车载道路测试确认了实时性能（延迟99毫秒）和成功的城市部署。通过将可解释的推理与精确控制结合起来，AR1展示了通往4级自动驾驶的实际路径。我们计划在未来的更新中发布AR1模型和部分CoC数据集。",
      "paper_summary": {
        "summary": "NVIDIA's Alpamayo-R1 (AR1) introduces a vision-language-action model that integrates causally-grounded reasoning with trajectory planning for autonomous driving. This approach enhances performance and safety in complex long-tail scenarios, achieving real-time inference and improved decision-making consistency.",
        "originalProblem": [
          "End-to-end autonomous driving systems often lack the explicit high-level reasoning needed to robustly handle complex and safety-critical \"long-tail\" scenarios.",
          "Existing vision-language-action (VLA) models for autonomous driving frequently suffer from a lack of explicit, causally-grounded reasoning or inconsistencies between their generated explanations and actual driving actions.",
          "Achieving strong generalization, interpretability, and verifiable decision-making remains a challenge for current autonomous driving systems aiming for Level 4 autonomy."
        ],
        "solution": [
          "Alpamayo-R1 (AR1) is a modular VLA architecture that integrates a structured Chain of Causation (CoC) reasoning framework with trajectory planning.",
          "A novel CoC dataset, generated via a hybrid auto-labeling and human-in-the-loop pipeline, provides decision-grounded and causally-linked reasoning traces.",
          "A multi-stage training strategy is employed, involving action modality injection, supervised fine-tuning on the CoC dataset, and RL-based post-training to refine reasoning and enforce consistency."
        ],
        "keyInsights": [
          "Causally-grounded and structurally aligned reasoning is critical for robust autonomous driving, enabling better utilization of contextual information and improved handling of ambiguous long-tail scenarios.",
          "A multi-stage training approach, particularly with Reinforcement Learning (RL) post-training guided by reasoning quality and consistency rewards, is effective in aligning high-level reasoning with low-level control actions.",
          "Combining a powerful, domain-specific VLM backbone (like NVIDIA's Cosmos-Reason) with an efficient continuous action decoder (e.g., flow-matching) facilitates real-time, physically feasible, and generalizable autonomous driving."
        ],
        "results": [
          "AR1 achieved a 12% reduction in minADE₆@6s (0.868m) in challenging scenarios and improved the overall AlpaSim score from 0.38 to 0.50, demonstrating enhanced safety and robustness in simulations.",
          "RL post-training improved reasoning quality by approximately 45% and increased reasoning-action consistency by 37%, ensuring generated explanations accurately reflect predicted actions.",
          "The model demonstrated real-time inference with a latency of 99ms on NVIDIA RTX 6000 Pro Blackwell hardware and successfully navigated complex urban environments in on-vehicle road tests."
        ]
      },
      "image_url": "image/2511.00088v1.png",
      "universal_paper_id": "2511.00088",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 78,
          "last_7_days": 78
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-30T01:25:34.000Z",
      "publication_date": "2025-10-30T01:25:34.000Z",
      "updated_at": "2025-11-04T05:40:09.701Z",
      "topics": [
        "agents",
        "autonomous-vehicles",
        "causal-inference",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "cs.RO",
        "deep-reinforcement-learning",
        "generative-models",
        "imitation-learning",
        "reasoning",
        "robotic-control",
        "vision-language-models"
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
      "abstract": "二次成本的缩放点积注意力是将自回归语言模型扩展到长上下文的一个主要障碍。线性时间的注意力和状态空间模型（SSM）提供了可扩展的替代方案，但通常限制于一阶或基于核的近似方法，这可能限制了表现力。我们介绍了高阶线性注意力（HLA），这是一种因果流机制，通过紧凑的前缀充要统计量实现更高层次的交互。在二阶情况下，HLA 维护一个恒定大小的状态，并在不实际化任何 $n \\times n$ 矩阵的情况下以线性时间计算每个标记的输出。我们给出了封闭形式的流标识、使用两个额外摘要的严格因果掩码变体，以及基于关联扫描的块并行训练方案，准确重现串行递归的激活。我们还概述了对三阶及更高阶的扩展。总体而言，这些结果将 HLA 定位为一个有原则的、可扩展的构建模块，结合了类似注意力的、数据依赖的混合与现代递归架构的效率。项目页面：此链接。",
      "paper_summary": null,
      "image_url": "image/2510.27258v1.png",
      "universal_paper_id": "2510.27258",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 93,
          "last_7_days": 93
        },
        "public_total_votes": 15
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
      "id": "019a4d5f-1c61-767e-a36d-7673ef03b2d4",
      "paper_group_id": "019a4d5f-1c61-767e-a36d-7673ef03b2d4",
      "title": "Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning",
      "abstract": "测试时强化学习（TTRL）提供了一种无标签的范式，仅在推理时使用合成信号来适应模型，但其成功取决于构造可靠的学习信号。标准的方法如多数投票常常会崩溃到虚假的但流行的答案。我们介绍了自我和谐（Self-Harmony），这是一个建立在简单直觉上的框架：正确的答案在原始问题和其意译之间应该保持稳定。自我和谐通过在两个互补角色中使用单一模型来实现这一点：解答者（Solver）生成答案，重述者（Reframer）对输入进行改写。在此基础上，我们进一步提出了一种伪标签方法：它不是采用多数投票，而是通过使用调和平均数来汇总这些原始和重述视图中的答案频率。这一过程自然选择了在重述下稳定的解决方案，从而避免了偏向于依赖视图的虚假答案的常见陷阱。至关重要的是，这不需要人类监督或辅助模型。在各种推理基准测试中，自我和谐在无标签的测试时设置中实现了最先进的结果，在多种方法中在30个设置中的28个中排名第一。除了准确性外，它还表现出前所未有的稳健性，在所有实验中都没有出现训练失败，强调了其稳定性和可靠性。",
      "paper_summary": null,
      "image_url": "image/2511.01191v1.png",
      "universal_paper_id": "2511.01191",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 20,
          "last_7_days": 20
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-11-03T03:34:34.000Z",
      "publication_date": "2025-11-03T03:34:34.000Z",
      "updated_at": "2025-11-04T05:37:50.177Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "reasoning",
        "reinforcement-learning",
        "self-supervised-learning",
        "test-time-inference"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4f7f-4789-7eec-a646-5d4b67231260",
      "paper_group_id": "019a4f7f-4789-7eec-a646-5d4b67231260",
      "title": "TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images Reasoning",
      "abstract": "视觉推理的前沿正在转向像OpenAI o3这样的模型，这些模型能够智能地创建和操作工具，以图像转换为问题解决方案，这也被称为链式思维中的“以图像思维”。然而，现有的基准测试未能完全捕捉到这一先进能力。即使是视觉搜索，当前以图像思维方法最常用的基准，只测试基础操作，如定位和裁剪，几乎无法提供对更复杂、动态和依赖工具的推理的深入洞察。我们引入了\\textbf{TIR-Bench}，这是一个全面的基准，旨在评估在13个不同任务中进行的代理性以图像思维的能力，每个任务都需要在链式思维中对图像处理和操作的新工具使用。我们评估了22个多模态大型语言模型（MLLM），从领先的开源模型到明确增强工具使用的专有模型。结果表明，TIR-Bench普遍具有挑战性，强大的表现需要真正的以图像思维能力。最后，我们呈现了一项比较直接微调与代理性微调的初步研究。",
      "paper_summary": {
        "summary": "TIR-Bench introduces a comprehensive benchmark to evaluate advanced agentic visual reasoning in multimodal large language models (MLLMs). It comprises 13 diverse tasks requiring active image manipulation and programmatic tool use, revealing that state-of-the-art MLLMs achieve only up to 46% accuracy and underscoring the necessity of agentic capabilities.",
        "originalProblem": [
          "Existing MLLM evaluation benchmarks are primarily confined to textual reasoning or static visual understanding, failing to assess active interaction with visual information.",
          "Current benchmarks for agentic MLLMs offer a narrow view, focusing mostly on visual search tasks and overlooking complex, dynamic, and tool-dependent visual manipulations like image rotation, contrast enhancement, or object reassembly.",
          "A lack of comprehensive benchmarks exists to properly evaluate MLLMs capable of 'thinking-with-images,' where models actively manipulate and modify visual data as part of their problem-solving chain-of-thought."
        ],
        "solution": [
          "Introduces TIR-Bench, a comprehensive benchmark featuring 13 diverse tasks specifically designed to necessitate active, tool-based visual reasoning and image manipulation beyond static analysis.",
          "Compiles a dataset of 1215 examples, generated through a hybrid approach of new creation/annotation, curation from existing datasets, and programmatic generation, to ensure task diversity and minimize data contamination.",
          "Evaluates 22 leading MLLMs, including open-source, proprietary, and agentic tool-using models, in a zero-shot setting to assess their capabilities in agentic visual reasoning.",
          "Conducts specific experiments to assess function-calling proficiency and compares the efficacy of direct supervised fine-tuning versus agentic fine-tuning for visual operation tasks."
        ],
        "keyInsights": [
          "TIR-Bench is exceptionally challenging for all evaluated models, with the highest average performance being only 46%, indicating significant room for improvement in MLLM visual reasoning.",
          "Agentic tool-using capabilities are crucial for complex visual reasoning, as models without explicit tool integration perform substantially worse than those that can actively manipulate images.",
          "Current agentic MLLMs still exhibit limitations in the depth of tool integration, particularly in calling sophisticated external specialized tools beyond basic image processing primitives for tasks like object segmentation.",
          "Agentic fine-tuning, which trains models on full problem-solving trajectories including intermediate image manipulations, is more effective than direct supervised fine-tuning for tasks requiring visual operations."
        ],
        "results": [
          "The top-performing model, o3-TU, achieved an average accuracy of 46% on TIR-Bench, outperforming the best non-tool-using MLLM (Gemini-2.5-pro) by nearly 17% and its non-tool-using counterpart (o3) by 19%.",
          "Traditional, non-agentic MLLMs performed poorly, often scoring slightly above random guess levels, confirming that static visual analysis is insufficient for TIR-Bench tasks.",
          "A pilot study on fine-tuning showed that Tool-Use SFT consistently and significantly outperformed Direct SFT on the Rotated Image OCR task, with Direct SFT showing no positive performance scaling with increased data size.",
          "Function-calling experiments demonstrated that explicit prompting strategies significantly improve MLLM performance in utilizing function-calling capabilities, and newer models like o3 make more iterative function calls per problem."
        ]
      },
      "image_url": "image/2511.01833v1.png",
      "universal_paper_id": "2511.01833",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 15,
          "last_7_days": 15
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-11-03T18:40:17.000Z",
      "publication_date": "2025-11-03T18:40:17.000Z",
      "updated_at": "2025-11-04T15:32:12.809Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.CV",
        "fine-tuning",
        "multi-modal-learning",
        "tool-use",
        "vision-language-models",
        "visual-reasoning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 13,
      "github_url": "https://github.com/agents-x-project/TIR-Bench",
      "distance": 1
    },
    {
      "id": "019a4efc-66cb-77e4-8048-875970ee585a",
      "paper_group_id": "019a4efc-66cb-77e4-8048-875970ee585a",
      "title": "Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning",
      "abstract": "最近，先进的大型语言模型（LLM）以越来越快的速度出现。然而，当面临复杂问题时，大多数用户往往无法提供准确有效的提示来与LLM互动，从而限制了LLM的性能。为了解决这一挑战，我们提出了Prompt-R1，这是一种端到端的强化学习框架，利用小规模的LLM与大规模LLM协作，替代用户互动以更好地解决问题。这种协作被视为多轮提示互动，小规模LLM进行思考并生成提示，而大规模LLM则执行复杂推理。设计了一种双重约束奖励，以优化正确性、生成质量和推理准确性。Prompt-R1提供了一个即插即用的框架，支持与各种大规模LLM的推理和训练。对多个公开数据集的实验表明，Prompt-R1在各项任务中显著优于基线模型。我们的代码可以在此https URL上公开获取。",
      "paper_summary": null,
      "image_url": "image/2511.01016v1.png",
      "universal_paper_id": "2511.01016",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 14,
          "last_7_days": 14
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-11-02T17:11:03.000Z",
      "publication_date": "2025-11-02T17:11:03.000Z",
      "updated_at": "2025-11-04T13:09:15.595Z",
      "topics": [
        "Computer Science",
        "cs.CL"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 10,
      "github_url": "https://github.com/QwenQKing/Prompt-R1",
      "distance": 1
    },
    {
      "id": "019a4d2d-6f03-7507-864d-832d78690de2",
      "paper_group_id": "019a4d2d-6f03-7507-864d-832d78690de2",
      "title": "Information-theoretic minimax and submodular optimization algorithms for multivariate Markov chains",
      "abstract": "我们研究了有限的多元马尔可夫链在 $d$ 维乘积态空间上的信息论最小极大问题。给定一组 $\\mathcal B=\\{P_1,\\ldots,P_n\\}$ 的 $\\pi$-平稳转移矩阵和一类由坐标集 $[d]$ 的分割 $\\mathbf S$ 所诱导的可分解模型 $\\mathcal F = \\mathcal{F}(\\mathbf{S})$，我们试图通过分析 $$\\min_{Q\\in\\mathcal F}\\max_{P\\in\\mathcal B} D_{\\mathrm{KL}}^{\\pi}(P\\|Q),$$ 来最小化最坏情况下的信息损失，其中 $D_{\\mathrm{KL}}^{\\pi}(P\\|Q)$ 是从 $Q$ 到 $P$ 的 $\\pi$ 加权 KL 散度。我们通过强对偶性和我们推导的毕达哥拉斯恒等式，将上述最小极大问题重新表述为对 $n$-概率单纯形的凹最大化。这使我们能够构造一个信息论博弈，并证明混合策略纳什均衡总是存在；并提出一种投影子梯度算法来近似解决该最小极大问题，并提供可验证的保证。通过将最小极大问题转化为 $\\mathbf{S}$ 中的象限次模函数，这激励我们考虑一个最大最小最大次模优化问题，并研究一种两层子梯度贪婪过程以近似解决这一广义问题。对于库里-韦斯和伯努利-拉普拉斯模型的马尔可夫链的数值实验展示了这些提出的算法的实用性，并揭示了这些例子中的稀疏最优结构。",
      "paper_summary": null,
      "image_url": "image/2511.00769v1.png",
      "universal_paper_id": "2511.00769",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 18,
          "last_7_days": 18
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-11-02T02:33:55.000Z",
      "publication_date": "2025-11-02T02:33:55.000Z",
      "updated_at": "2025-11-04T04:43:34.531Z",
      "topics": [
        "Mathematics",
        "math.OC",
        "math.PR",
        "stat.CO",
        "Statistics"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a4d92-4bb5-702b-9c3f-1adb3d571e87",
      "paper_group_id": "019a4d92-4bb5-702b-9c3f-1adb3d571e87",
      "title": "LongCat-Flash-Omni Technical Report",
      "abstract": "我们推出了LongCat-Flash-Omni，这是一款具有5600亿参数的先进开源全模态模型，在实时音视频互动中表现出色。通过采用一种灵感来自课程的渐进式训练策略，逐步过渡到从简单到越来越复杂的模态序列建模任务，LongCat-Flash-Omni在保持强大单模态能力的同时，达到了全面的多模态能力。在此基础上，LongCat-Flash采用了高性能的Shortcut连接混合专家（MoE）架构，具备零计算专家，LongCat-Flash-Omni集成了高效的多模态感知和语音重建模块。尽管具有560亿参数的庞大规模（其中27亿被激活），LongCat-Flash-Omni仍然实现了低延迟的实时音视频互动。为了训练基础设施，我们开发了一种模态解耦并行方案，专门设计用于管理大规模多模态训练中固有的数据和模型异构性。这种创新方法通过维持超过90%的文本训练吞吐量，表现出卓越的效率。广泛的评估表明，LongCat-Flash-Omni在开源模型中的全模态基准测试中取得了最先进的性能。此外，它在文本、图像和视频理解以及音频理解和生成等多种模态特定任务中也实现了具有高度竞争力的结果。我们提供了模型架构设计、训练过程和数据策略的全面概述，并将模型开源，以促进社区未来的研究和发展。",
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
      "image_url": "image/2511.00279v1.png",
      "universal_paper_id": "2511.00279",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 41,
          "last_7_days": 41
        },
        "public_total_votes": 9
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
      "github_stars": 317,
      "github_url": "https://github.com/meituan-longcat/LongCat-Flash-Omni",
      "distance": 1
    }
  ],
  "page": 0
};