const papersData = {
  "papers": [
    {
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们推出了Tongyi DeepResearch，一个具有自主性的、专门为长期、大规模信息搜索研究任务设计的大型语言模型。为了激励自主的深度研究能力，Tongyi DeepResearch通过一个端到端的训练框架开发，该框架结合了自主的中期训练和自主的后期训练，使得在复杂任务中进行可扩展的推理和信息搜寻成为可能。我们设计了一个高可扩展性的数据合成管道，完全自动化，无需依赖昂贵的人为标注，支持所有训练阶段。通过为每个阶段构建定制化环境，我们的系统实现了稳健和一致的交互。Tongyi DeepResearch拥有305亿个参数，每个令牌仅激活33亿个参数，在一系列自主深度研究基准测试中表现出色，包括人类的最后考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES和xbench-DeepSearch-2510。我们将模型、框架和完整解决方案开源，以赋能社区。",
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
        "total_votes": 13,
        "visits_count": {
          "all": 407,
          "last_7_days": 407
        },
        "public_total_votes": 41
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
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLMs）推理能力方面展现了显著的潜力。然而，RL在LLMs上的成功在很大程度上依赖于人工策划的数据集和可验证的奖励，这限制了它们的可扩展性和普遍性。近期的自我对弈强化学习方法，受益于这一范式在游戏和围棋中的成功，旨在提升LLM的推理能力而无需人工标注的数据。然而，这些方法主要依赖于一个有反馈的基础环境（例如，Python解释器或游戏引擎）；将其扩展到一般领域仍然面临挑战。为了解决这些问题，我们提出了多智能体进化（MAE）框架，它使LLM能够在解决多样化任务（包括数学、推理和一般知识问答）中自我进化。MAE的核心设计基于三个相互作用的智能体（提问者、求解者、评判者），这些智能体由单个LLM实例化，并应用强化学习来优化它们的行为。提问者生成问题，求解者尝试解决方案，评判者在共同进化的同时评估两者。对Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试中实现了平均4.54%的提升。这些结果突显了MAE作为一种可扩展、数据高效的方法，能够在最小依赖人工策划监督的情况下提升LLM的普遍推理能力。",
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
        "total_votes": 7,
        "visits_count": {
          "all": 260,
          "last_7_days": 260
        },
        "public_total_votes": 30
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
      "abstract": "关于大规模监督微调AI代理的公共研究成果仍然相对较少，因为收集代理训练数据面临独特的挑战。在这项工作中，我们认为瓶颈不在于缺乏基础数据源，而在于各种数据在异构格式、工具和接口中碎片化。为此，我们提出了代理数据协议（ADP），这是一种轻量级的表示语言，充当不同格式的代理数据集与统一的代理训练管道之间的“中介语言”。ADP的设计足够表达各种任务，包括API/工具使用、浏览、编码、软件工程和一般代理工作流程，同时在解析和训练时无需针对每个数据集进行工程处理。在实验中，我们将13个现有的代理训练数据集统一为ADP格式，并将标准化的ADP数据转换为多个代理框架的训练准备格式。我们在这些数据上进行了监督微调（SFT），并展示了相较于相应基线模型平均约20%的性能提升，在标准编码、浏览、工具使用和研究基准上达到了最先进或接近最先进的性能，而无需领域特定的调优。所有代码和数据均已公开发布，希望ADP能够帮助降低标准化、可扩展和可重复的代理训练的门槛。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 131,
          "last_7_days": 131
        },
        "public_total_votes": 14
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
      "id": "019a315b-20e8-7941-a237-262a8aeeb18c",
      "paper_group_id": "019a315b-20e8-7941-a237-262a8aeeb18c",
      "title": "An efficient probabilistic hardware architecture for diffusion-like models",
      "abstract": "概率人工智能的普及促进了专用随机计算机的提案。尽管这些提案显示出潜在的效率提升，但由于依赖基本有限的建模技术和奇特的、无法扩展的硬件，它们未能获得 traction。本文针对这些缺点，提出了一种全晶体管的概率计算机，能够在硬件层面实现强大的去噪模型。系统级分析表明，基于我们架构的设备在一个简单的图像基准测试中，可以在使用约1/10,000的能量的情况下，实现与GPU的性能平衡。",
      "paper_summary": {
        "summary": "Extropic Corporation presents a probabilistic hardware architecture that integrates Denoising Thermodynamic Models (DTMs) with a CMOS-compatible, all-transistor random number generator, achieving an estimated 10,000-fold energy reduction compared to GPU-based diffusion models for generative tasks.",
        "originalProblem": [
          "Large-scale AI systems, particularly LLMs, face an unsustainable energy consumption trajectory, with current GPU-centric hardware leading to immense power footprints.",
          "Previous probabilistic computing efforts suffered from the \"mixing-expressivity tradeoff\" (MET) in monolithic Energy-Based Models (EBMs) and relied on non-scalable \"exotic\" hardware components.",
          "The \"Hardware Lottery\" phenomenon has locked AI algorithm development into suboptimal hardware architectures, hindering the exploration of more energy-efficient paradigms."
        ],
        "solution": [
          "Introduces Denoising Thermodynamic Models (DTMs) which overcome the MET by sequentially composing simpler, hardware-implementable EBMs for gradual denoising.",
          "Develops a Denoising Thermodynamic Computer Architecture (DTCA) featuring an all-transistor random number generator (RNG) that is CMOS-compatible, fast, and energy-efficient.",
          "Proposes a system-level co-design of algorithms and hardware, enabling massively parallel arrays of primitive sampling circuitry optimized for Boltzmann machines."
        ],
        "keyInsights": [
          "DTMs bypass the mixing-expressivity tradeoff by breaking down complex generative tasks into a chain of simpler conditional sampling problems, each handled by an efficient EBM.",
          "An all-transistor RNG based on subthreshold transistor networks provides a scalable, energy-efficient, and commercially viable stochastic primitive for probabilistic hardware, avoiding exotic components.",
          "Hybrid Thermodynamic-Deterministic Machine Learning (HTDML) offers a pathway to combine the energy efficiency of probabilistic hardware with the expressivity of classical neural networks for complex tasks."
        ],
        "results": [
          "Projects a 10,000-fold reduction in energy consumption for generative tasks (e.g., binarized Fashion-MNIST) compared to GPU-based diffusion models, while achieving performance parity.",
          "Demonstrates an all-transistor RNG with a 100 ns autocorrelation decay time, ~350 aJ/bit energy consumption, and a small ~3x3 µm footprint, validated for CMOS compatibility.",
          "Shows DTMs can be trained stably using an Adaptive Correlation Penalty (ACP), maintaining well-mixed sampling chains and improving monotonically in quality, unlike traditional monolithic EBMs."
        ]
      },
      "image_url": "image/2510.23972v1.png",
      "universal_paper_id": "2510.23972",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 121,
          "last_7_days": 121
        },
        "public_total_votes": 12
      },
      "first_publication_date": "2025-10-28T01:09:19.000Z",
      "publication_date": "2025-10-28T01:09:19.000Z",
      "updated_at": "2025-10-29T19:04:07.144Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "energy-efficient-ml",
        "generative-models",
        "hardware-aware-algorithms",
        "image-generation",
        "inference-optimization",
        "ml-systems"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "paper_group_id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "title": "DeepAgent: A General Reasoning Agent with Scalable Toolsets",
      "abstract": "大型推理模型已经显示出强大的问题解决能力，然而现实世界的任务通常需要外部工具和长时间的互动。现有的代理框架通常遵循预定义的工作流程，这限制了自主和全局任务的完成。在本文中，我们介绍了DeepAgent，这是一种端到端的深度推理代理，能够在单一、一致的推理过程中执行自主思考、工具发现和行动执行。为了应对长时间交互的挑战，特别是来自多个工具调用的上下文长度爆炸和交互历史的累积，我们引入了一种自主记忆折叠机制，将过去的交互压缩成结构化的情节记忆、工作记忆和工具记忆，减少错误的累积，同时保留关键信息。为了高效且稳定地教授通用工具的使用，我们开发了一种端到端的强化学习策略，即ToolPO，利用LLM模拟的API，并将工具调用优势归因应用于赋予工具调用令牌细粒度的信用。在八个基准上的大量实验，包括一般工具使用任务（ToolBench、API-Bank、TMDB、Spotify、ToolHop）和下游应用（ALFWorld、WebShop、GAIA、HLE），表明DeepAgent在标记工具和开放集合工具检索场景中始终优于基线。这项工作朝着实现更通用和更强大代理的方向迈出了一步，以便应用于现实世界。代码和演示可在此HTTPS链接获取。",
      "paper_summary": {
        "summary": "DeepAgent is presented as an end-to-end reasoning agent that leverages large language models for autonomous thinking, dynamic tool discovery, and action execution from scalable toolsets. The framework significantly outperforms previous workflow-based methods, achieving up to 36.4% higher success rates on general tool-use benchmarks and state-of-the-art results on complex downstream applications, supported by novel memory management and a tailored reinforcement learning strategy.",
        "originalProblem": [
          "Existing LLM agents are limited by rigid, predefined workflows that constrain their autonomy and ability to adapt dynamically to diverse tasks.",
          "Current frameworks typically rely on restricted and often small toolsets, preventing agents from addressing the vast and varied requirements of real-world scenarios.",
          "Long-horizon interactions pose challenges due to context length limitations and the accumulation of errors without effective mechanisms for self-correction or memory management.",
          "Training general-purpose tool-using agents is inefficient and unstable, primarily due to costly, latent real-world API interactions and the difficulty in providing fine-grained feedback for intermediate tool calls."
        ],
        "solution": [
          "Developed DeepAgent, an end-to-end deep reasoning agent that unifies autonomous thinking, dynamic tool discovery, and action execution within a single, coherent reasoning process.",
          "Introduced an autonomous memory folding mechanism, supported by a brain-inspired memory schema (episodic, working, and tool memory), to effectively manage long-horizon interactions by compressing history and enabling strategic reconsideration.",
          "Proposed ToolPO, an end-to-end reinforcement learning strategy that uses LLM-simulated APIs for stable training and incorporates fine-grained tool-call advantage attribution for precise credit assignment during policy optimization."
        ],
        "keyInsights": [
          "Unifying thinking, dynamic tool discovery, and action execution into a continuous reasoning stream allows LLMs to maintain a global perspective on tasks, moving beyond fragmented 'Reason-Act-Observe' cycles.",
          "Effective memory management, via autonomous folding and a structured, brain-inspired memory schema, is critical for handling long-horizon interactions, preventing context overflow, and enabling strategic replanning.",
          "Reinforcement learning for tool-using agents can be stabilized and made efficient by utilizing LLM-simulated APIs and providing precise, localized rewards for intermediate tool invocations."
        ],
        "results": [
          "DeepAgent achieved superior performance on general tool usage tasks, outperforming the best 32B baselines by up to 36.4% on TMDB (89.0% vs 55.0%) and 22.8% on Spotify (75.4% vs 52.6%).",
          "On complex downstream applications, DeepAgent attained state-of-the-art performance among 32B models, scoring 53.3 on GAIA (compared to HiRA's 42.5) and a 91.8% success rate on ALFWorld (versus HiRA's 84.3%).",
          "Ablation studies confirmed the criticality of ToolPO training and autonomous memory folding, as their removal led to significant performance declines, highlighting their essential contributions to robust agent behavior."
        ]
      },
      "image_url": "image/2510.21618v1.png",
      "universal_paper_id": "2510.21618",
      "metrics": {
        "total_votes": 30,
        "visits_count": {
          "all": 1403,
          "last_7_days": 1403
        },
        "public_total_votes": 92
      },
      "first_publication_date": "2025-10-24T16:24:01.000Z",
      "publication_date": "2025-10-24T16:24:01.000Z",
      "updated_at": "2025-10-27T04:20:03.854Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.IR",
        "cs.LG",
        "deep-reinforcement-learning",
        "fine-tuning",
        "reasoning",
        "reinforcement-learning",
        "tool-use",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Renmin University of China",
          "image": "images/organizations/renmin.png"
        },
        {
          "name": "Xiaohongshu Inc.",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 25,
      "github_url": "https://github.com/RUC-NLPIR/DeepAgent",
      "distance": 1
    },
    {
      "id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "paper_group_id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "title": "Scaling Latent Reasoning via Looped Language Models",
      "abstract": "现代大型语言模型（LLM）主要通过显式的文本生成进行“思考”，例如链式推理（CoT），这使得推理过程推迟到后期训练，并未充分利用预训练数据。我们提出并开源了Ouro，名为递归的乌罗波罗斯，属于一类预训练的循环语言模型（LoopLM），该模型通过以下方式将推理融入预训练阶段：(i) 潜在空间中的迭代计算，(ii) 用于学习深度分配的熵正则化目标，以及 (iii) 扩展到7.7万亿个标记。Ouro 1.4B和2.6B模型在多个基准测试中表现优越，性能与高达12B的最先进大型语言模型相匹配。通过控制实验，我们证明了这一优势并非来自于知识容量的增加，而是来自于更出色的知识操作能力。我们还展示了LoopLM所产生的推理痕迹与最终输出的对齐程度优于显式的CoT。我们希望我们的结果展示了LoopLM作为推理时代一种新颖扩展方向的潜力。我们的模型可以在此http URL找到。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 102,
          "last_7_days": 102
        },
        "public_total_votes": 13
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
      "id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "paper_group_id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "title": "The Principles of Diffusion Models",
      "abstract": "本专著介绍了指导扩散模型发展的核心原理，追溯其起源，并展示出如何从共享的数学思想中产生多样化的形式。扩散建模首先通过定义一个向前过程，将数据逐渐腐蚀为噪声，将数据分布与一个简单先验通过一系列中间分布相连接。目标是学习一个逆过程，将噪声转化回数据，同时恢复相同的中间状态。我们描述了三个互补的视角。变分视角受到变分自编码器的启发，将扩散视为学习逐步去除噪声。基于评分的视角根植于基于能量的建模，学习不断变化的数据分布的梯度，指示如何将样本推向更可能的区域。基于流的视角与归一化流有关，将生成视为遵循一个平滑路径，通过学习的速度场将样本从噪声转移到数据。这些观点有一个共同的基础：一个时间依赖的速度场，其流动将一个简单的先验传输到数据。采样则相当于解决一个微分方程，该方程沿着连续轨迹将噪声演变为数据。在此基础上，专著讨论了可控生成的指导、高效数值求解器以及基于扩散的流图模型，这些模型学习在任意时间之间的直接映射。它为具备基本深度学习知识的读者提供了对扩散模型的概念性和数学基础的理解。",
      "paper_summary": {
        "summary": "Authored by leading researchers from Sony AI, OpenAI, and Stanford, this monograph synthesizes the rapidly evolving field of diffusion models by clarifying their theoretical foundations and unifying diverse formulations into a single continuous-time generative framework. It systematically covers the origins, unifies variational, score-based, and flow-based perspectives, and outlines advancements in sampling and generation techniques.",
        "originalProblem": [
          "The rapid proliferation of diverse diffusion model formulations (e.g., DDPM, NCSN, Flow Matching) led to a fragmented literature.",
          "Researchers and practitioners found it challenging to grasp the underlying mathematical connections and theoretical coherence across different approaches.",
          "A lack of a systematic and unified understanding hindered entry into the field and efficient progress in research."
        ],
        "solution": [
          "The monograph provides a comprehensive and authoritative review, acting as a pedagogical guide for the field of diffusion models.",
          "It rigorously demonstrates the mathematical equivalence of variational, score-based, and flow-based approaches under a continuous-time framework.",
          "It clarifies the theoretical underpinnings, including the central roles of the score function, SDEs, ODEs, and the Fokker-Planck equation, and systematically covers practical advancements like guidance and efficient sampling."
        ],
        "keyInsights": [
          "Variational, score-based, and flow-based diffusion models are mathematically equivalent formulations of a single continuous-time generative process, all implicitly or explicitly learning a time-dependent vector field.",
          "The 'conditioning trick,' which transforms intractable marginal objectives into tractable conditional ones, is a foundational enabler for training across all major diffusion model formulations.",
          "The generation process can be fundamentally understood as solving a differential equation (SDE or Probability Flow ODE), providing a principled basis for analyzing and accelerating sampling."
        ],
        "results": [
          "The monograph conclusively demonstrates the mathematical equivalence between DDPM, NCSN/Score SDE, and Flow Matching paradigms, showing they are manifestations of a unified continuous-time generative process.",
          "It establishes the Fokker-Planck equation as the universal underlying law governing the evolution of marginal probability densities in diffusion models, regardless of stochastic or deterministic dynamics.",
          "The work systematically categorizes and explains advancements in diffusion models, including guidance mechanisms for controllable generation, training-free acceleration methods (numerical solvers), and training-based acceleration techniques (distillation, learning from scratch like Consistency Models)."
        ]
      },
      "image_url": "image/2510.21890v1.png",
      "universal_paper_id": "2510.21890",
      "metrics": {
        "total_votes": 23,
        "visits_count": {
          "all": 668,
          "last_7_days": 668
        },
        "public_total_votes": 56
      },
      "first_publication_date": "2025-10-24T02:29:02.000Z",
      "publication_date": "2025-10-24T02:29:02.000Z",
      "updated_at": "2025-10-29T01:33:04.144Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.GR",
        "cs.LG",
        "generative-models",
        "image-generation",
        "optimization-methods",
        "representation-learning",
        "statistical-learning",
        "unsupervised-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 3,
      "github_url": "https://github.com/Shiying-Zhang/diffusion-theory-discussion",
      "distance": 1
    },
    {
      "id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "paper_group_id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "title": "Uniform Discrete Diffusion with Metric Path for Video Generation",
      "abstract": "连续空间视频生成迅速推进，而离散方法由于误差积累和长期上下文不一致而滞后。在这项工作中，我们重新审视了离散生成建模，并提出了均匀分布的扩散与度量路径（URSA），这是一个简单而强大的框架，弥合了可扩展视频生成与连续方法之间的差距。URSA的核心是将视频生成任务表述为对离散时空代币的迭代全局精细化。它整合了两个关键设计：线性度量路径和与分辨率相关的时间步移转机制。这些设计使URSA能够有效扩展到高分辨率图像合成和长时段视频生成，同时所需的推理步骤显著减少。此外，我们引入了一种异步时间微调策略，统一了单一模型中的多种任务，包括插值和图像转视频生成。在具有挑战性的视频和图像生成基准上的大量实验表明，URSA始终优于现有的离散方法，并且其性能与最新的连续扩散方法相当。代码和模型可以在此链接找到。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 149,
          "last_7_days": 149
        },
        "public_total_votes": 16
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
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于大语言模型的网络代理在信息检索中展现出巨大的潜力，但它们在长时间任务中的有效性受到上下文管理的基本权衡的制约。现有的ReAct-based代理由于累积了嘈杂的原始历史而遭遇上下文饱和，而在每个步骤固定总结整个历史的方法则面临关键细节不可逆转损失的风险。为了解决这些问题，我们引入了AgentFold，这是一种新颖的代理范式，专注于主动的上下文管理，受到人类认知过程中的回顾性整合的启发。AgentFold将其上下文视为一个动态的认知工作空间，主动进行塑造，而不是被动地填充记录。在每个步骤中，它学习执行“折叠”操作，以在多个层面上管理其历史轨迹：它可以进行细致的浓缩，以保留重要的细微细节，或者进行深度整合，以抽象出整个多步骤的子任务。在显著的基准测试中，结果令人瞩目：通过简单的监督微调（无需持续预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上达到了36.2%，在BrowseComp-ZH上达到了47.3%。值得注意的是，这一性能不仅超越或匹配了大规模开源模型，如DeepSeek-V3.1-671B-A37B，还超越了领先的专有代理，比如OpenAI的o4-mini。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 119,
          "last_7_days": 119
        },
        "public_total_votes": 14
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
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我改善系统需要与环境互动以实现持续适应。我们介绍了SPICE（自我对弈语料环境），这是一个强化学习框架，单一模型担任两个角色：挑战者从大型语料库中提取文档以生成多样化的推理任务，和推理者来解决这些任务。在对抗性动态中，挑战者在推理者能力的前沿创建了一个自动课程，而语料库的基础提供了丰富的、几乎无尽的外部信号，支持持续改进。与现有的无基础自我对弈方法相比，这些方法的收益更为有限，SPICE在多个模型系列的数学 (+8.9%) 和一般推理 (+9.8%) 基准测试中实现了一致的提升。我们的分析揭示了文档基础在SPICE中是一个关键要素，它不断生成自身愈发挑战的目标并实现这些目标，从而实现持续的自我改善。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 95,
          "last_7_days": 95
        },
        "public_total_votes": 13
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
      "id": "019a29d3-5ad4-745c-89dd-4b5f2a90d263",
      "paper_group_id": "019a29d3-5ad4-745c-89dd-4b5f2a90d263",
      "title": "A Survey of Data Agents: Emerging Paradigm or Overstated Hype?",
      "abstract": "大型语言模型（LLMs）的快速发展催生了数据代理的出现——这些自主系统旨在协调数据与人工智能生态系统，以解决复杂的数据相关任务。然而，“数据代理”这一术语目前存在定义模糊和不一致采用的问题，将简单的查询响应者与复杂的自主架构混为一谈。这种术语模糊性导致了用户期望的不匹配、责任挑战以及行业增长的障碍。受德尔福自动驾驶标准SAE J3016的启发，本调查引入了首个系统性的层次分类法，用于数据代理，包含六个级别，明确划分并追踪从手动操作（L0）到生成的、完全自主的数据代理（L5）之间的渐进性变化，从而澄清能力边界和责任分配。通过这一视角，我们提供了一个结构化的现有研究回顾，按照自主性逐渐递增的顺序，涵盖了用于数据管理、准备和分析的专门数据代理，以及朝着更具自主性的多功能综合系统的发展努力。我们进一步分析了推进数据代理所需的关键演变飞跃和技术差距，特别是正在进行的L2到L3过渡，即数据代理从程序执行演变为自主编排。最后，我们以一个前瞻性的路线图结束，展望主动生成的数据代理的到来。",
      "paper_summary": {
        "summary": "This paper presents the inaugural systematic, hierarchical taxonomy for data agents, classifying their autonomy from Level 0 to Level 5. The framework addresses current terminological ambiguity in the field and provides a structured overview of existing LLM-powered data systems and a roadmap for future research across the data lifecycle.",
        "originalProblem": [
          "Significant terminological ambiguity and inconsistent definitions surrounding the term \"data agent,\" leading to varied interpretations of system capabilities.",
          "Resulting risks for users (mismatched expectations), governance (unclear accountability), and industry (hindered growth and overstated claims).",
          "Absence of a standardized, systematic framework to classify data agent autonomy and capabilities."
        ],
        "solution": [
          "Introduced a novel, systematic hierarchical taxonomy for data agents (L0-L5 autonomy), drawing inspiration from the SAE J3016 standard for driving automation.",
          "Categorized existing research and industrial data agent systems within this taxonomy, focusing on their capabilities across data management, preparation, and analysis.",
          "Analyzed the \"evolutionary leaps\" required to progress between autonomy levels, detailing technical challenges and paradigm shifts."
        ],
        "keyInsights": [
          "Most current data agents operate at L1 (preliminary assistance) or L2 (partial autonomy with environmental perception), demonstrating a 'glass ceiling' in comprehensive task orchestration.",
          "Achieving L3 (conditional autonomy) requires overcoming limitations in pipeline orchestration, comprehensive data lifecycle coverage, advanced strategic reasoning, and adaptation to dynamic environments.",
          "The proposed taxonomy provides a common language for researchers and practitioners, enabling objective comparison and clearer understanding of data agent capabilities and limitations."
        ],
        "results": [
          "Successfully established the first systematic, hierarchical taxonomy (L0-L5) for data agents, offering a standardized vocabulary for the field.",
          "Mapped the current state-of-the-art, showing significant development in L1 and L2 systems and identifying early 'Proto-L3' efforts in both academia and industry.",
          "Delineated critical technical gaps and a clear roadmap for future research, particularly highlighting the challenges in transitioning from L2 to L3 autonomy."
        ]
      },
      "image_url": "image/2510.23587v1.png",
      "universal_paper_id": "2510.23587",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 161,
          "last_7_days": 161
        },
        "public_total_votes": 17
      },
      "first_publication_date": "2025-10-27T17:54:07.000Z",
      "publication_date": "2025-10-27T17:54:07.000Z",
      "updated_at": "2025-10-28T07:58:28.564Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.DB",
        "data-curation",
        "generative-models",
        "ml-systems",
        "reasoning",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 58,
      "github_url": "https://github.com/HKUSTDial/awesome-data-agents",
      "distance": 1
    },
    {
      "id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "paper_group_id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "title": "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation",
      "abstract": "最近大型语言模型（LLMs）的成功重新引发了人们对推荐系统是否能实现类似扩展收益的兴趣。传统推荐系统以庞大的嵌入表为主，随着嵌入维度的增加，其性能往往趋于平稳。相比之下，新兴的生成范式用自回归变换器生成的紧凑语义标识（SID）序列替代了嵌入。然而，大多数工业部署仍然是专有的，这留下了两个基本问题： (1) 预期的扩展法则在公共基准上是否成立？ (2) 什么是实现竞争性能的最低后训练方案？\n我们提出了MiniOneRec，尽我们所知，这是第一个完全开源的生成推荐框架，提供了一个涵盖SID构建、监督微调和面向推荐的强化学习的端到端工作流程。我们通过残差量化变分自编码器生成SIDs，并对0.5B到7B参数范围内的Qwen骨干网络进行了后训练，数据集为Amazon Review。我们的实验揭示了随着模型尺寸的增加，训练和评估损失均显示出一致的下降趋势，验证了生成方法的参数效率。为了进一步提升性能，我们提出了一种轻量且有效的后训练管道，该管道 (1) 强制执行全流程SID对齐，(2) 应用带约束解码和混合奖励的强化学习。结合这些技术，显著提高了排名准确性和候选多样性。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 96,
          "last_7_days": 96
        },
        "public_total_votes": 15
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
    },
    {
      "id": "019a2df4-d452-707d-b69b-aa09894baaa3",
      "paper_group_id": "019a2df4-d452-707d-b69b-aa09894baaa3",
      "title": "Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs",
      "abstract": "尽管多模态大型语言模型（MLLMs）在视觉理解方面表现出色，但在需要视觉规划和想象的复杂场景中常常遇到困难。受到人类如何利用草图作为一种视觉思维形式来发展和传达思想的启发，我们提出了潜在草图板（Latent Sketchpad），这一框架为MLLMs配备了一个内部视觉便签本。传统上，MLLMs的内部视觉表征仅限于感知理解。我们重新利用它们以支持生成性视觉思维，而不影响推理能力。基于前沿的MLLMs，我们的方法将视觉生成直接集成到它们本土的自回归推理过程中。这使得模型能够将文本推理与视觉潜象的生成交替进行。这些潜象指导内部思维过程，并可以转换为草图图像以便于理解。为实现这一目标，我们引入了两个组件：一个上下文感知的视觉头自回归地产生视觉表征，预训练的草图解码器将其渲染为人类可理解的图像。我们在新的数据集MazePlanning上评估了该框架。在对各种MLLMs的实验中，结果显示潜在草图板提供的推理性能与其主干模型相当，甚至优于主干模型。它进一步在不同的前沿MLLMs中泛化，包括Gemma3和Qwen2.5-VL。通过将模型的文本推理扩展到视觉思维，我们的框架为更丰富的人机交互和更广泛的应用开辟了新机会。更多细节和资源请访问我们的项目页面：这个https URL。",
      "paper_summary": {
        "summary": "A framework called Latent Sketchpad extends pretrained multimodal large language models by enabling them to generate and integrate abstract visual thoughts as sketches directly into their reasoning process. This modular approach enhances performance on complex visual planning tasks and provides interpretable visual traces of the model's internal thinking.",
        "originalProblem": [
          "Multimodal Large Language Models (MLLMs) struggle with complex visual reasoning and planning, often lacking dynamic visual imagination.",
          "Current MLLMs are primarily designed for visual understanding, but lack the capacity to actively generate or manipulate internal visual representations to aid their reasoning process dynamically.",
          "Existing solutions rely on external tools with predefined capabilities or unified generative models that prioritize pixel-level realism and require extensive architectural modifications or retraining."
        ],
        "solution": [
          "Latent Sketchpad introduces a modular framework to equip pretrained MLLMs with an internal visual scratchpad, comprising a Context-Aware Vision Head and a Pretrained Sketch Decoder.",
          "The framework enables MLLMs to autoregressively interleave textual reasoning steps with the generation of abstract visual latents, allowing these latents to guide and inform internal thought processes.",
          "A Pretrained Sketch Decoder translates these internal visual latents into human-interpretable sketch images, offering a transparent view into the model's evolving reasoning trajectory."
        ],
        "keyInsights": [
          "MLLMs can effectively leverage abstract visual thoughts (sketches) in their latent space to improve complex planning and spatial reasoning, a more suitable approach than generating photorealistic images for reasoning tasks.",
          "A plug-and-play modular architecture allows existing, powerful MLLM backbones (e.g., Gemma3, Qwen2.5-VL) to gain generative visual thinking capabilities without requiring extensive retraining or fundamental architectural changes.",
          "Providing human-interpretable visual traces of an MLLM's internal reasoning enhances transparency and offers new avenues for debugging and human-AI collaboration."
        ],
        "results": [
          "Latent Sketchpad substantially improves the performance of proprietary MLLMs (e.g., GPT-4o) on complex MAZEPLANNING tasks, transforming success rates from below 20% to competitive levels by providing crucial spatial cues.",
          "The framework demonstrates broad applicability, integrating successfully with diverse pretrained backbones like Gemma3 and Qwen2.5-VL, consistently enhancing their multimodal reasoning capacity.",
          "Generated visual thoughts maintain high structural stability and layout consistency (e.g., >99% Layout Consistency Rate for Gemma3+LS) and actively correlate with enhanced task success rates (e.g., Gemma3+LS Visual Success Rate reaches 75.6% versus 70% for its text-only baseline)."
        ]
      },
      "image_url": "image/2510.24514v1.png",
      "universal_paper_id": "2510.24514",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 68,
          "last_7_days": 68
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-28T15:26:20.000Z",
      "publication_date": "2025-10-28T15:26:20.000Z",
      "updated_at": "2025-10-29T03:13:31.218Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "cs.CV",
        "explainable-ai",
        "generative-models",
        "human-ai-interaction",
        "image-generation",
        "multi-modal-learning",
        "reasoning",
        "representation-learning",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "University of Cambridge",
          "image": "images/organizations/university-of-cambridge.svg+xml"
        },
        {
          "name": "Chinese Academy of Sciences",
          "image": "images/organizations/chinese-academy-of-sciences.jpeg"
        },
        {
          "name": "Nanjing University",
          "image": "images/organizations/nanjing.png"
        },
        {
          "name": "Microsoft",
          "image": "images/organizations/microsoft.png"
        },
        {
          "name": "Chinese Academy of Sciences Institute of Automation",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 6,
      "github_url": "https://github.com/hwanyu112/Latent-Sketchpad",
      "distance": 1
    },
    {
      "id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "paper_group_id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "title": "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations",
      "abstract": "人类通过多感官的协同作用学习抽象概念，一旦形成，这种表征通常可以通过单一模态回忆起来。受到这一原则的启发，我们提出了Concerto，这是一种简约的空间认知人类概念学习模拟，结合了3D同模自蒸馏与2D-3D跨模态联合嵌入。尽管其简单，Concerto却学习到了更连贯和信息量更丰富的空间特征，通过零样本可视化得到了验证。它在3D场景感知的线性探测中，分别比单独的最先进2D和3D自我监督模型提升了14.2%和4.8%，以及比它们的特征连接效果更佳。在完全微调后，Concerto在多个场景理解基准测试中设立了新的最先进结果（例如，ScanNet上的80.7% mIoU）。我们还展示了一个为视频提升点云空间理解量身定制的Concerto变体，以及一个将Concerto表征线性投影到CLIP语言空间的转换器，从而实现开放世界感知。这些结果凸显了Concerto在空间表征中展现了卓越的细粒度几何和语义一致性。",
      "paper_summary": {
        "summary": "Concerto, a joint 2D-3D self-supervised learning framework developed by researchers from The University of Hong Kong, The Chinese University of Hong Kong, and Harbin Institute of Technology, synergistically combines intra-modal 3D self-distillation and cross-modal 2D-3D joint embedding prediction. This approach learns unified spatial representations, achieving 80.7% mIoU for 3D semantic segmentation on ScanNet and demonstrating robust data efficiency.",
        "originalProblem": [
          "Representations learned independently from 2D images and 3D point clouds do not fully overlap, missing complementary spatial information.",
          "3D self-supervised learning has historically lagged behind its 2D counterpart due to the sparse, unstructured, and often incomplete nature of point cloud data.",
          "Existing multi-modal approaches often rely on simple feature concatenation or one-way distillation, failing to achieve true synergistic integration of modalities."
        ],
        "solution": [
          "Concerto employs a dual-branch self-supervised learning framework combining 3D intra-modal self-distillation (extending Sonata) and 2D-3D cross-modal joint embedding prediction.",
          "The 3D branch refines point cloud representations using a teacher-student clustering objective, while the 2D-3D branch trains the 3D encoder to predict features from a frozen 2D image encoder (DINOv2) using camera parameters for correspondence.",
          "This approach enables a 'multisensory synergy' where modalities continuously inform and enhance each other, leading to emergent, unified spatial representations."
        ],
        "keyInsights": [
          "Deep, joint self-supervised learning between 2D images and 3D point clouds can yield representations that are fundamentally richer and more coherent than those derived from individual modalities or simple fusions.",
          "A 'minimalist simulation of human concept learning' through combined intra-modal refinement and cross-modal prediction fosters emergent spatial cognition in machines.",
          "The learned spatial representations can be effectively aligned with human language via linear projection into a language space like CLIP, opening avenues for open-world 3D perception."
        ],
        "results": [
          "Concerto achieved 77.3% mIoU for 3D semantic segmentation on ScanNet with linear probing, surpassing concatenated SOTA 2D and 3D features by 1.4%.",
          "With full fine-tuning, it established new state-of-the-art with 80.7% mIoU on ScanNet and 39.2% on ScanNet200.",
          "The framework demonstrated superior data efficiency, outperforming supervised methods across all protocols in the ScanNet Data Efficient benchmark, especially in data-limited scenarios."
        ]
      },
      "image_url": "image/2510.23607v1.png",
      "universal_paper_id": "2510.23607",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 182,
          "last_7_days": 182
        },
        "public_total_votes": 23
      },
      "first_publication_date": "2025-10-27T17:59:59.000Z",
      "publication_date": "2025-10-27T17:59:59.000Z",
      "updated_at": "2025-10-28T03:49:52.885Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "geometric-deep-learning",
        "multi-modal-learning",
        "object-detection",
        "representation-learning",
        "self-supervised-learning",
        "semantic-segmentation",
        "transfer-learning",
        "vision-language-models",
        "zero-shot-learning"
      ],
      "organization_info": [
        {
          "name": "The Chinese University of Hong Kong",
          "image": "images/organizations/chinesehongkong.png"
        },
        {
          "name": "The University of Hong Kong",
          "image": "images/organizations/hku.png"
        },
        {
          "name": "Harbin Institute of Technology (Shenzhen)",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 58,
      "github_url": "https://github.com/Pointcept/Concerto",
      "distance": 1
    },
    {
      "id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "paper_group_id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "title": "Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents",
      "abstract": "我们提出了Game-TARS，一个通用游戏代理，使用与人类对齐的原生键盘鼠标输入建立的统一、可扩展的动作空间进行训练。与基于API或GUI的方法不同，这种范式允许在异构领域（包括操作系统、网络和模拟游戏）进行大规模的连续预训练。Game-TARS在超过500B的多样轨迹和多模态数据上进行了预训练。关键技术包括逐渐减小的连续损失，以减少因果混淆，以及一种高效的稀疏思维策略，平衡推理深度和推理成本。实验表明，Game-TARS在开放世界Minecraft任务上实现了约2倍于之前最佳模型的成功率，在未见过的网络3D游戏中接近新手人类的普遍性，并在FPS基准测试中超越了GPT-5、Gemini-2.5-Pro和Claude-4-Sonnet。在训练和测试时间的扩展结果中，证实了统一动作空间在跨游戏和多模态数据的扩展中持续保持改善。我们的结果表明，简单、可扩展的动作表示与大规模预训练相结合，为具有广泛计算机使用能力的通用代理提供了一个有希望的路径。",
      "paper_summary": {
        "summary": "Game-TARS develops a generalist multimodal game agent using a human-native keyboard-mouse interaction paradigm for scalable, continual pre-training. The agent achieves superior performance in diverse unseen game environments like Minecraft, web 3D games, and FPS benchmarks, often outperforming existing specialized and general-purpose models.",
        "originalProblem": [
          "Traditional game AI agents suffer from highly customized, environment-specific action spaces that severely limit their generalization to new domains.",
          "Existing generalist agents often rely on environment-specific APIs or Graphical User Interface elements, creating a semantic gap and hindering true universality across digital environments.",
          "Integrating sophisticated reasoning efficiently into agent behavior without incurring prohibitively high computational costs remains a challenge."
        ],
        "solution": [
          "Introduces a \"Human-Native Interaction paradigm\" that defines a universal, low-level action space based on keyboard and mouse inputs, enabling massive cross-domain pre-training.",
          "Employs \"native Sparse ReAct pretraining\" using an online \"think-aloud\" protocol and visual anchors for causal alignment, generating reasoning only at critical decision points.",
          "Utilizes a history-aware decaying loss function during continual pre-training to re-weight repetitive actions, compelling the model to focus on high-entropy decision boundaries."
        ],
        "keyInsights": [
          "A universal, low-level keyboard-mouse action space provides a scalable foundation for generalist agents, confirming that simple, general representations effectively scale with data and compute.",
          "Selectively incorporating reasoning through a Sparse Thinking strategy significantly improves efficiency and performance in complex tasks by balancing deliberation with action.",
          "Mitigating imbalanced action distributions with a decaying loss function during pre-training yields more robust and diverse learned behaviors by preventing the model from exploiting dataset biases."
        ],
        "results": [
          "Game-TARS-MoE-mini achieved success rates of 72.0% in Embodied, 55.4% in GUI, and 66.1% in Combat tasks on the unseen Minecraft MCU benchmark, substantially outperforming state-of-the-art baselines.",
          "Demonstrated competitive generalization against human players and surpassed GPT-5 in unseen web 3D games, also outperforming GPT-5, Claude-4-Sonnet, and Gemini-2.5-Pro in the FPS Vizdoom benchmark.",
          "The Sparse Thinking strategy improved Minecraft success rate to 63% while reducing average token consumption by 45% compared to greedy thinking, optimizing reasoning efficiency."
        ]
      },
      "image_url": "image/2510.23691v1.png",
      "universal_paper_id": "2510.23691",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 78,
          "last_7_days": 78
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-27T17:43:51.000Z",
      "publication_date": "2025-10-27T17:43:51.000Z",
      "updated_at": "2025-10-29T08:13:32.278Z",
      "topics": [
        "agents",
        "Computer Science",
        "continual-learning",
        "cs.AI",
        "deep-reinforcement-learning",
        "imitation-learning",
        "inference-optimization",
        "multi-modal-learning",
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
      "id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "paper_group_id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "title": "Real Deep Research for AI, Robotics and Beyond",
      "abstract": "随着人工智能和机器人研究的快速增长，每年产生超过一万篇论文，研究人员保持更新变得越来越困难。快速发展的趋势、跨学科工作的兴起以及探索超出自身专业领域的必要性都加剧了这一挑战。为了解决这些问题，我们提出了一种通用的流程，能够系统地分析任何研究领域：识别新兴趋势、发掘跨领域机会，并为新研究提供具体的起点。在这项工作中，我们提出了“真实深度研究”（RDR），这是一个应用于人工智能和机器人领域的综合框架，特别关注基础模型和机器人技术的进展。我们还简要扩展了对其他科学领域的分析。主要论文详细介绍了RDR流程的构建，而附录则提供了每个分析主题的广泛结果。我们希望这项工作能为在人工智能及相关领域工作的研究人员提供启示。",
      "paper_summary": {
        "summary": "Researchers from UC San Diego, NVIDIA, META, UW-Madison, and UNC developed Real Deep Research (RDR), a generalizable pipeline that systematically analyzes scientific literature using off-the-shelf large language and multimodal models. This framework creates high-quality, structured surveys, identifies research trends, and uncovers cross-domain opportunities, consistently outperforming commercial LLMs in expert evaluations.",
        "originalProblem": [
          "The exponential growth of scientific literature, particularly in AI and robotics, leads to information overload, making it difficult for researchers to stay informed and identify new directions.",
          "Traditional expert-written survey papers are resource-intensive and quickly become outdated, while existing automated LLM-based tools often lack the necessary domain expertise and can produce superficial or hallucinated analyses.",
          "Researchers struggle to efficiently identify emerging trends, recognize interdisciplinary connections, and quickly grasp new topics outside their immediate expertise."
        ],
        "solution": [
          "The Real Deep Research (RDR) pipeline was developed, utilizing off-the-shelf LLMs/LMMs (e.g., Doubao, o3, NV-Embed-v2) in a four-stage process: Data Preparation, Content Reasoning, Content Projection, and Embedding Analysis.",
          "Domain experts define specific perspectives (e.g., Input, Modeling, Output for Foundation Models) to guide LMMs in extracting structured, granular information from papers.",
          "Extracted natural language descriptions are projected into an informative latent embedding space, which is then clustered to group similar papers and generate descriptive keywords for survey construction and advanced analysis."
        ],
        "keyInsights": [
          "Integrating expert-defined analytical perspectives with the reasoning capabilities of LMMs allows for deep, structured understanding of scientific literature, reducing hallucination and improving survey quality.",
          "Transforming detailed paper information into a semantic embedding space enables robust clustering, visualization of research trends over time, and the discovery of cross-domain connections through knowledge graphs.",
          "A modular pipeline approach, relying on high-performing, pre-trained foundation models without additional training, provides a scalable and generalizable solution for comprehensive scientific literature analysis across diverse research areas."
        ],
        "results": [
          "RDR achieved the highest overall performance in expert evaluations for survey quality, with an average rank of 1.30, consistently outperforming commercial LLMs such as GPT5 and Gemini in most categories.",
          "The framework's underlying `nvidia/NV-Embed-v2` embeddings demonstrated state-of-the-art unsupervised clustering performance, achieving 84.86% accuracy on AG News and 52.91% on 20 News Groups, validating the quality of its latent space representation.",
          "RDR successfully identified and visualized emerging research trends (e.g., \"teleoperation\" and \"dexterous manipulation\" in robotics) and mapped cross-domain knowledge graphs, effectively highlighting interdisciplinary opportunities and high-impact papers."
        ]
      },
      "image_url": "image/2510.20809v1.png",
      "universal_paper_id": "2510.20809",
      "metrics": {
        "total_votes": 29,
        "visits_count": {
          "all": 1716,
          "last_7_days": 1716
        },
        "public_total_votes": 117
      },
      "first_publication_date": "2025-10-23T17:59:05.000Z",
      "publication_date": "2025-10-23T17:59:05.000Z",
      "updated_at": "2025-10-24T16:11:10.700Z",
      "topics": [
        "clustering-algorithms",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "human-ai-interaction",
        "information-extraction",
        "ml-systems",
        "recommender-systems",
        "text-classification"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 1,
      "github_url": "https://github.com/realdeepresearch/realdeepresearch.github.io",
      "distance": 1
    },
    {
      "id": "019a330f-ab28-7eba-9849-ff7b835837c9",
      "paper_group_id": "019a330f-ab28-7eba-9849-ff7b835837c9",
      "title": "The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution",
      "abstract": "现实世界中的语言代理必须处理跨多个应用程序的复杂多步骤工作流。例如，一个代理可能通过与日历和文件系统协调来管理电子邮件，或者监控生产数据库以检测异常并生成遵循操作手册的报告。然而，现有的语言代理基准通常侧重于狭窄的领域或简化的任务，缺乏评估代理在现实世界表现所需的多样性、真实性和长期复杂性。为了解决这一问题，我们推出了工具十项全能赛（称为Toolathlon），这是一个为语言代理提供多样化应用和工具、真实环境设置以及可靠执行评估的基准。Toolathlon涵盖32个软件应用和604个工具，范围从Google日历和Notion等日常平台到WooCommerce、Kubernetes和BigQuery等专业平台。大多数工具都是基于我们可能已修订或自实施的高质量模型上下文协议（MCP）服务器。与之前的工作主要确保功能真实性但环境状态多样性有限不同，我们提供了来自真实软件的真实初始环境状态，例如有数十名学生的Canvas课程或真实的财务电子表格。该基准总共包括108个手动来源或制作的任务，平均需要与多个应用程序交互约20轮才能完成。每个任务都可以通过专门的评估脚本严格验证。对最先进模型的综合评估突出显示了它们的显著不足：表现最好的模型Claude-4.5-Sonnet的成功率仅为38.6%，平均调用工具的轮次为20.2，而顶级开放权重模型DeepSeek-V3.2-Exp的成功率达到20.1%。我们期望Toolathlon能够推动更强大的语言代理的发展，以执行现实世界中的长期任务。",
      "paper_summary": {
        "summary": "A new benchmark, TOOLATHLON, has been introduced to rigorously evaluate language agents on their ability to execute diverse, realistic, and long-horizon tasks across 32 real-world applications. Evaluations using this benchmark reveal that even leading state-of-the-art models achieve low success rates, with the top-performing Claude-4.5-Sonnet reaching only 38.6% Pass@1.",
        "originalProblem": [
          "Existing language agent benchmarks often use narrow domains, simplified tasks, or artificial environments, failing to capture real-world complexity.",
          "Many benchmarks rely on synthetic data, mocked tools, or subjective LLM judges, leading to unreliable and unreproducible evaluations.",
          "Current evaluation methodologies do not adequately test agents' capabilities in long-horizon tasks requiring multi-step planning and cross-application orchestration."
        ],
        "solution": [
          "Developed TOOLATHLON, a comprehensive benchmark with 108 tasks across 32 real-world applications and 604 tools, spanning seven diverse domains.",
          "Designed tasks to be long-horizon (average 20 interaction turns), multi-application, and featuring realistic initial environment states and \"fuzzy\" instructions.",
          "Implemented a strictly verifiable, execution-based evaluation framework using deterministic scripts to check final environment states within isolated containerized or remote real-world environments."
        ],
        "keyInsights": [
          "Current state-of-the-art language agents, including top proprietary models, demonstrate significant limitations in reliably executing complex, real-world, long-horizon tasks.",
          "Agents struggle with long-context scenarios, particularly processing and extracting information from overlong tool outputs, leading to reduced success rates.",
          "There is a notable disparity between models' occasional success (Pass@3) and consistent performance (Pass^3), highlighting a lack of robustness and reliability."
        ],
        "results": [
          "The best-performing proprietary model, Claude-4.5-Sonnet, achieved only a 38.6% Pass@1 success rate across all tasks.",
          "Leading open-source models showed a noticeable performance gap, with DeepSeek-V3.2-Exp reaching a 20.1% Pass@1 success rate.",
          "Hallucinating non-existent tool names proved more detrimental to task success than tool execution errors that provided feedback for agent adjustment."
        ]
      },
      "image_url": "image/2510.25726v1.png",
      "universal_paper_id": "2510.25726",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 39,
          "last_7_days": 39
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-29T17:32:49.000Z",
      "publication_date": "2025-10-29T17:32:49.000Z",
      "updated_at": "2025-10-30T03:00:56.232Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "multi-task-learning",
        "reasoning",
        "tool-use"
      ],
      "organization_info": [
        {
          "name": "Carnegie Mellon University",
          "image": "images/organizations/cmu.jpg"
        },
        {
          "name": "Duke University",
          "image": "images/organizations/duke-university.jpeg"
        },
        {
          "name": "HKUST",
          "image": "images/organizations/hkust.jpg"
        },
        {
          "name": "All Hands AI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 17,
      "github_url": "https://github.com/hkust-nlp/Toolathlon",
      "distance": 1
    },
    {
      "id": "019a2ebc-20e5-784d-83f2-c72225fc1637",
      "paper_group_id": "019a2ebc-20e5-784d-83f2-c72225fc1637",
      "title": "WebLeaper: Empowering Efficiency and Efficacy in WebAgent via Enabling Info-Rich Seeking",
      "abstract": "基于大型语言模型（LLM）的智能体已经成为解决开放性问题的一种变革性方法，其中信息获取（IS）是一项核心能力，能够实现自主推理和决策。尽管以往的研究主要集中在提高检索深度上，我们观察到当前的IS智能体往往存在搜索效率低下的问题，这反过来限制了整体表现。导致这种低效率的一个关键因素是训练任务中目标实体的稀疏性，这限制了智能体学习和推广高效搜索行为的机会。为了解决这些挑战，我们提出了WebLeaper，一个构建高覆盖率IS任务和生成高效解决方案轨迹的框架。我们将IS形式化为一个树形结构的推理问题，使得在有限上下文中可以嵌入更大规模的目标实体。利用精心整理的维基百科表格，我们提出了三种合成IS任务的变体：基础型、联合型和反联合型，旨在系统性地提高IS的效率和有效性。最后，我们通过保留那些同时准确且高效的训练轨迹来策划训练过程，确保模型在正确性和搜索性能上都得到优化。在五个IS基准测试（BrowserComp、GAIA、xbench-DeepSearch、WideSearch和Seal-0）上进行的基础和综合设置的广泛实验表明，我们的方法在有效性和效率上始终优于强大的基线。",
      "paper_summary": {
        "summary": "Alibaba Group's Tongyi Lab developed WebLeaper, a framework that enhances the efficiency and efficacy of LLM-based web agents in information seeking by synthesizing entity-intensive tasks and guiding trajectory construction. The system consistently achieved state-of-the-art results among open-source agents, delivering performance comparable to proprietary models like Claude-4-Sonnet on challenging benchmarks, including a 73.2 score on GAIA and 38.8 on BrowseComp.",
        "originalProblem": [
          "LLM-based information-seeking agents often exhibit low search efficiency, characterized by redundant queries and retrieval of irrelevant information, leading to inflated computational and time costs.",
          "This inefficiency largely stems from the sparsity of target entities in conventional training tasks, which provides insufficient opportunities for agents to learn and generalize efficient search behaviors within a constrained context.",
          "Existing research primarily focused on improving retrieval depth or correctness, often overlooking the critical aspect of search efficiency in agent performance."
        ],
        "solution": [
          "WebLeaper introduces a novel data synthesis framework for creating \"entity-intensive\" information-seeking tasks that contain a significantly larger number of target entities, modeled as tree-structured reasoning problems from curated Wikipedia tables.",
          "It employs an information-guided trajectory construction process that filters agent-generated solutions based on both Information-Seeking Rate (completeness) and Information-Seeking Efficiency (minimal actions per entity) to ensure high-quality, optimal training sequences.",
          "A hybrid reward system, incorporating a granular F-score for entity-intensive tasks, is integrated for reinforcement learning fine-tuning to provide stable and nuanced training signals for both efficacy and efficiency."
        ],
        "keyInsights": [
          "Training agents with entity-intensive tasks, particularly those leveraging \"Union\" and \"Reverse-Union\" structures to combine or reverse reasoning flows across multiple sources, significantly enhances agents' planning and decision-making capabilities for complex information gathering.",
          "Simultaneously optimizing for both information completeness (ISR) and efficiency (ISE) during trajectory generation is crucial for curating high-quality, concise, and goal-directed solution paths for complex web browsing tasks.",
          "A hybrid reward system, capable of providing granular feedback on entity retrieval, effectively guides reinforcement learning to achieve joint gains in both agent effectiveness and search efficiency, demonstrating their complementary nature."
        ],
        "results": [
          "WebLeaper consistently achieved state-of-the-art performance among open-source agents, with its comprehensive training setting reaching scores such as 73.2 on GAIA, 38.8 on BrowseComp, and 72.0 on xbench-DeepSearch.",
          "The \"Reverse-Union\" task synthesis variant yielded the strongest performance improvements, averaging +4.34 over the \"Union\" variant, by enhancing an agent's planning and multi-step reasoning capabilities.",
          "The framework demonstrated joint improvements in both effectiveness (higher accuracy/scores) and efficiency (fewer average action rounds) across all evaluated benchmarks, validating the core hypothesis that efficient search leads to better overall performance with reduced operational costs."
        ]
      },
      "image_url": "image/2510.24697v1.png",
      "universal_paper_id": "2510.24697",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 45,
          "last_7_days": 45
        },
        "public_total_votes": 8
      },
      "first_publication_date": "2025-10-28T17:51:42.000Z",
      "publication_date": "2025-10-28T17:51:42.000Z",
      "updated_at": "2025-10-29T06:51:12.485Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.CL",
        "data-curation",
        "fine-tuning",
        "reasoning",
        "tool-use",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Alibaba Group",
          "image": "images/organizations/alibaba.png"
        },
        {
          "name": "Peking University",
          "image": "images/organizations/peking.png"
        },
        {
          "name": "Tongyi Lab",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/Alibaba-NLP/DeepResearch",
      "distance": 1
    },
    {
      "id": "019a32af-ef69-779e-a3cd-c0e8197ca80e",
      "paper_group_id": "019a32af-ef69-779e-a3cd-c0e8197ca80e",
      "title": "RegionE: Adaptive Region-Aware Generation for Efficient Image Editing",
      "abstract": "最近，基于指令的图像编辑（IIE）受到了广泛关注。在实际应用中，IIE往往只修改图像的特定区域，而其余区域基本保持不变。尽管这两种区域在生成难度和计算冗余性上有显著差异，但现有的IIE模型并未考虑这种区别，而是在整个图像上应用统一的生成过程。这激励我们提出了RegionE，一个自适应的、区域感知的生成框架，可以在不额外训练的情况下加速IIE任务。具体而言，RegionE框架由三个主要组成部分构成：1）自适应区域划分。我们观察到未编辑区域的轨迹是直的，允许在一次步骤中推断出多步去噪预测。因此，在去噪的早期阶段，我们根据最终估计结果与参考图像之间的差异，将图像划分为编辑和未编辑区域。2）区域感知生成。区分区域后，我们将未编辑区域的多步去噪替换为一步预测。对于编辑区域，轨迹是弯曲的，需要局部迭代去噪。为了提高局部迭代生成的效率和质量，我们提出了区域指令KV缓存，它在结合全局信息的同时降低计算成本。3）自适应速度衰减缓存。我们观察到编辑区域的相邻时间步表现出强烈的速度相似性，因此进一步提出了自适应速度衰减缓存，以加速局部去噪过程。我们将RegionE应用于最先进的IIE基础模型，包括Step1X-Edit、FLUX.1 Kontext和Qwen-Image-Edit。RegionE实现了2.57、2.41和2.06的加速因子。通过GPT-4o的评估证实，语义和感知的保真度得到了良好保持。",
      "paper_summary": {
        "summary": "RegionE is a training-free framework that accelerates instruction-based image editing (IIE) models by exploiting spatial and temporal redundancies. It achieves end-to-end speedups of 2-2.5x on state-of-the-art IIE models like Step1X-Edit, FLUX.1 Kontext, and Qwen-Image-Edit, while maintaining high image quality with PSNR values above 30dB and extremely low perceptual differences (LPIPS around 0.05).",
        "originalProblem": [
          "Existing Instruction-Based Image Editing (IIE) models incur high computational costs and inference latency, limiting real-time applications.",
          "Current IIE models process entire images uniformly, applying the same computationally intensive denoising steps to both edited and unedited regions, leading to spatial redundancy.",
          "Diffusion models exhibit temporal redundancy in attention layer Key/Value pairs and velocity outputs between adjacent timesteps, particularly in the denoising-only paradigm of IIE."
        ],
        "solution": [
          "RegionE introduces a three-stage, training-free framework: Stabilization Stage (STS), Region-Aware Generation Stage (RAGS), and Smooth Stage (SMS).",
          "An Adaptive Region Partition (ARP) identifies edited and unedited regions based on one-step image estimation, allowing for differential processing.",
          "Region-Instruction KV Cache (RIKVCache) optimizes edited region processing by reusing cached global context, and Adaptive Velocity Decay Cache (AVDCache) predicts velocities for temporal efficiency using learned decay factors."
        ],
        "keyInsights": [
          "Edited regions and unedited regions in IIE tasks exhibit fundamentally different generation trajectories during denoising, allowing for region-specific acceleration strategies.",
          "Key and Value pairs in Diffusion Transformer (DiT) attention layers, particularly for unedited and instruction tokens, remain highly stable across diffusion timesteps and can be effectively cached.",
          "Denoising velocities between consecutive timesteps in edited regions are highly similar in direction and exhibit predictable decay in magnitude, enabling adaptive velocity prediction."
        ],
        "results": [
          "RegionE demonstrated end-to-end speedups of 2.57x for Step1X-Edit, 2.41x for FLUX.1 Kontext, and 2.06x for Qwen-Image-Edit, significantly reducing inference latency.",
          "The framework maintained superior image quality, achieving high PSNR (e.g., 30.520 dB on Step1X-Edit) and low LPIPS scores (e.g., 0.054 on Step1X-Edit), consistently outperforming other acceleration methods by substantial margins.",
          "GPT-4o evaluations confirmed high semantic consistency and perceptual quality, with RegionE's outputs closely matching or slightly exceeding vanilla model performance, validating human-perceptible fidelity."
        ]
      },
      "image_url": "image/2510.25590v1.png",
      "universal_paper_id": "2510.25590",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 35,
          "last_7_days": 35
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-10-29T14:58:37.000Z",
      "publication_date": "2025-10-29T14:58:37.000Z",
      "updated_at": "2025-10-30T01:16:22.249Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "Imperial College London",
          "image": "images/organizations/imperial-college-london.jpeg"
        },
        {
          "name": "Fudan University",
          "image": "images/organizations/fudan-university.png"
        },
        {
          "name": "StepFun",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 6,
      "github_url": "https://github.com/Peyton-Chen/RegionE",
      "distance": 1
    },
    {
      "id": "019a2b6f-87f8-7da1-ae0b-feb3c5f7f47e",
      "paper_group_id": "019a2b6f-87f8-7da1-ae0b-feb3c5f7f47e",
      "title": "IGGT: Instance-Grounded Geometry Transformer for Semantic 3D Reconstruction",
      "abstract": "人类自然地将三维世界的几何结构和语义内容视为交织在一起的维度，从而能够对复杂场景进行连贯和准确的理解。然而，大多数先前的方法优先训练用于低级三维重建的大型几何模型，将高级空间理解孤立处理，忽视了这两种三维场景分析基本方面之间的重要相互作用，从而限制了推广性，并导致在下游三维理解任务中的表现不佳。最近的尝试通过简单地将三维模型与特定语言模型对齐来缓解这一问题，从而将感知限制在对齐模型的能力范围内，限制了对下游任务的适应性。在本文中，我们提出了实例基础几何变换器（InstanceGrounded Geometry Transformer，IGGT），这是一个端到端的大型统一变换器，用于统一空间重建和实例级上下文理解的知识。具体而言，我们设计了一种三维一致性对比学习策略，指导IGGT通过仅使用二维视觉输入来编码具有几何结构和实例基础聚类的统一表示。该表示支持将二维视觉输入一致地提升为具有明确不同对象实例的连贯三维场景。为便于完成这一任务，我们进一步构建了InsScene-15K，这是一个大型数据集，包含高质量的RGB图像、姿态、深度图以及三维一致性的实例级掩膜注释，并采用了新颖的数据整理流程。",
      "paper_summary": {
        "summary": "IGGT introduces an end-to-end framework for unified 3D reconstruction and instance-level contextual understanding, leveraging a large unified transformer and a new 3D-consistent dataset, InsScene-15K. The framework establishes new benchmarks across instance spatial tracking, open-vocabulary semantic segmentation, and QA scene grounding.",
        "originalProblem": [
          "Traditional 3D scene understanding approaches suffered from fragmented pipelines, leading to error propagation and neglected mutual context between geometry and semantics.",
          "Prior VLM-aligned methods faced issues like geometric detail over-smoothing, VLM vendor lock-in, and limited capability for fine-grained instance-level differentiation.",
          "A scarcity of large-scale, high-quality, 3D-consistent, instance-level annotated datasets hindered the training of robust foundation models."
        ],
        "solution": [
          "An Instance-Grounded Geometry Transformer (IGGT) unifies spatial reconstruction and instance-level understanding using a 1-billion parameter transformer with DINOv2 features and cross-modal fusion.",
          "3D-consistent contrastive supervision, combined with geometry supervision, encourages mutual enhancement between geometric and instance semantic learning.",
          "A new large-scale dataset, InsScene-15K, was created, featuring 3D-consistent instance-level masks across 15,000 scenes and 200 million images, to address data scarcity.",
          "An instance-grounded understanding paradigm uses generated 3D-consistent instance masks as a flexible 'plug-and-play' interface for various off-the-shelf Vision-Language Models (VLMs) and Large Multimodal Models (LMMs)."
        ],
        "keyInsights": [
          "Joint training of geometry and instance-level semantics within a unified transformer allows for mutual enhancement, leading to more coherent and accurate 3D scene understanding.",
          "Generating 3D-consistent instance masks provides a universal, flexible interface that decouples the 3D perception model from specific Vision-Language Models (VLMs), enabling 'plug-and-play' adaptability.",
          "The cross-modal fusion block is crucial for embedding fine-grained geometric awareness directly into instance representations, improving boundary precision and spatial sensitivity."
        ],
        "results": [
          "Achieved state-of-the-art performance in instance spatial tracking, with 69.41% Temporal mIoU on ScanNet and 73.02% Temporal mIoU on ScanNet++.",
          "Demonstrated leading capabilities in open-vocabulary semantic segmentation, outperforming prior methods by 8.34% mIoU in 2D and 4.97% mIoU in 3D on ScanNet++.",
          "Enabled accurate QA scene grounding by leveraging instance-grounded querying with Large Multimodal Models, showing superior multi-view consistency for complex prompts compared to vanilla LMMs."
        ]
      },
      "image_url": "image/2510.22706v1.png",
      "universal_paper_id": "2510.22706",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 79,
          "last_7_days": 79
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-26T14:57:44.000Z",
      "publication_date": "2025-10-26T14:57:44.000Z",
      "updated_at": "2025-10-28T15:28:40.952Z",
      "topics": [
        "Computer Science",
        "contrastive-learning",
        "cs.CV",
        "geometric-deep-learning",
        "image-segmentation",
        "neural-rendering",
        "representation-learning",
        "robotics-perception",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Northwestern Polytechnical University",
          "image": null
        },
        {
          "name": "Tsinghua University",
          "image": "images/organizations/tsinghua.png"
        },
        {
          "name": "The Chinese University of Hong Kong",
          "image": "images/organizations/chinesehongkong.png"
        },
        {
          "name": "Nanyang Technological University",
          "image": "images/organizations/nanyang-technological-university.png"
        },
        {
          "name": "StepFun, Inc.",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a3258-ba90-763b-b54f-f9819b50e6d9",
      "paper_group_id": "019a3258-ba90-763b-b54f-f9819b50e6d9",
      "title": "Generating Creative Chess Puzzles",
      "abstract": "尽管生成性人工智能在各个领域快速发展，但生成真正具有创意、美学和反直觉的输出仍然是一大挑战。本文提出了一种解决国际象棋难题领域相关困难的方法。我们首先对生成性人工智能架构进行基准测试，然后引入一个基于国际象棋引擎搜索统计的具有新颖奖励的强化学习框架，以克服一些不足。这些奖励旨在增强难题的独特性、反直觉性、多样性和现实感。我们的强化学习方法使反直觉难题的生成提高了10倍，从0.22%（监督学习）增至2.5%，超越了现有数据集的比率（2.1%）和最佳Lichess训练模型（0.4%）。我们的难题满足新颖性和多样性的基准，保持了美学主题，并且被人类专家评定为比现成书籍难题更具创意、趣味和反直觉性，甚至接近经典作品。我们最终的成果是一本策划的AI生成难题的小册子，得到了三位世界知名专家的创意认可。",
      "paper_summary": {
        "summary": "Google DeepMind, University of Oxford, and Mila researchers developed an AI system that generates creative and counter-intuitive chess puzzles by formalizing creativity metrics and employing a reinforcement learning framework. The system increased the generation rate of counter-intuitive puzzles to 2.5%, surpassing human-derived datasets, and produced puzzles rated by human experts as more creative and enjoyable than typical Lichess puzzles.",
        "originalProblem": [
          "Generating truly creative, aesthetic, and counter-intuitive content remains a significant challenge for AI, especially in domains requiring abstract reasoning.",
          "Existing chess puzzle datasets, like Lichess Puzzler, contain a scarcity of truly 'creative' or counter-intuitive puzzles, with only 2.1% meeting rigorous criteria.",
          "Traditional AI in games focuses primarily on performance, overlooking subjective qualities such as counter-intuitiveness and aesthetics, which are crucial for human enjoyment of puzzles."
        ],
        "solution": [
          "Formalized chess puzzle creativity by proposing computational metrics for uniqueness, novelty, and a novel measure of 'counter-intuitiveness' based on comparing shallow vs. deep chess engine evaluations.",
          "Utilized a reinforcement learning (RL) framework with a pre-trained generative AI model (auto-regressive transformer) to optimize directly for these newly defined creative attributes.",
          "Introduced diversity-filtering mechanisms, including KL divergence constraints, piece regularization, and intra/inter-batch novelty tests, to prevent 'entropy collapse' and ensure realism and diversity in generated puzzles."
        ],
        "keyInsights": [
          "Subjective creative attributes like 'counter-intuitiveness' can be effectively quantified through computational metrics, such as comparing chess engine evaluations at different search depths, enabling AI optimization.",
          "Reinforcement Learning, when combined with robust pre-training and mechanisms to promote diversity, can successfully guide generative models to produce complex and subjectively creative outputs, even when high-quality training data is scarce.",
          "Human expert evaluation plays a critical role in validating AI-generated creative content, demonstrating that AI can produce artifacts that resonate with human aesthetic appreciation and rival human compositions."
        ],
        "results": [
          "The RL framework dramatically increased the generation rate of counter-intuitive puzzles from a baseline of 0.22% (supervised transformer) to 2.5%, exceeding the 2.1% rate found in the Lichess training data.",
          "Implemented diversity filtering mechanisms successfully prevented 'entropy collapse,' leading to the continuous generation of novel and diverse puzzles, confirmed by board and principal variation distance metrics.",
          "Human chess experts rated the AI-generated puzzles as more creative, enjoyable, and counter-intuitive than typical Lichess puzzles, and comparable to human-composed book puzzles."
        ]
      },
      "image_url": "image/2510.23881v1.png",
      "universal_paper_id": "2510.23881",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 31,
          "last_7_days": 31
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-27T21:43:39.000Z",
      "publication_date": "2025-10-27T21:43:39.000Z",
      "updated_at": "2025-10-29T23:41:07.088Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "deep-reinforcement-learning",
        "fine-tuning",
        "generative-models",
        "human-ai-interaction",
        "optimization-methods",
        "reinforcement-learning",
        "synthetic-data",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "paper_group_id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "title": "Parallel Loop Transformer for Efficient Test-Time Computation Scaling",
      "abstract": "大型语言模型（LLMs）功能强大，但在推理时通常速度太慢且成本太高，无法满足实际应用需求。循环变压器通过在多个计算步骤或“循环”中重用相同权重来节省参数。然而，这种方法存在一个主要缺陷：循环依次运行，导致推理延迟和内存需求随着每个新增循环而增加。这使得它们不适合快速应用。为了解决这个问题，我们引入了并行循环变压器（PLT）。PLT是一种新架构，能够提供深度循环模型的性能优势，同时具备标准非循环模型的低延迟。PLT通过两项关键技术实现这一目标。首先，跨循环并行性（CLP）通过同时计算不同令牌的不同循环，打破了顺序依赖，所有操作在一次传递中完成。其次，为了防止内存成本增长，我们采用了一种高效表示增强策略。该方法将第一个循环的内存（KV缓存）与所有其他循环共享。然后，它使用门控滑动窗口注意力（G-SWA）将这一共享的全局信息与局部信息结合，维持高精度。我们的实验表明，PLT在与标准变压器相比几乎没有额外延迟或内存成本的情况下，达到了传统循环模型的高精度。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 31,
          "last_7_days": 31
        },
        "public_total_votes": 5
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
      "id": "019a3111-3414-7eea-aaee-8ec23cde4125",
      "paper_group_id": "019a3111-3414-7eea-aaee-8ec23cde4125",
      "title": "Transitive RL: Value Learning via Divide and Conquer",
      "abstract": "在这项工作中，我们提出了传递强化学习（TRL），这是一种基于分治范式的新值学习算法。TRL旨在解决离线目标条件强化学习（GCRL）问题，其目标是找到一种策略，能够以最少的步骤从任何状态到达任何其他状态。TRL将GCRL中存在的三角不等式结构转化为实用的分治值更新规则。这与其他值学习范式相比具有几个优势。与时间差分（TD）方法相比，TRL在偏差积累方面的影响较小，因为原则上它只需要$O(\\log T)$的递归（相比于TD学习中的$O(T)$）来处理长度为$T$的轨迹。与蒙特卡罗方法不同，TRL由于执行动态规划，因此在高方差方面的影响较小。实验表明，TRL在高度具有挑战性、长时间跨度的基准任务中，相比于以前的离线GCRL算法实现了最佳性能。",
      "paper_summary": {
        "summary": "Researchers at UC Berkeley introduce Transitive RL (TRL), a divide-and-conquer value learning algorithm for offline goal-conditioned reinforcement learning, which mitigates the \"curse of horizon\" by employing a transitive Bellman update with in-trajectory subgoals and expectile regression. This method achieves leading performance on long-horizon robotic tasks and competitive results across various standard benchmarks.",
        "originalProblem": [
          "Traditional temporal difference (TD) learning in offline goal-conditioned RL suffers from severe bias accumulation, leading to the \"curse of horizon\" in long-horizon tasks.",
          "Monte Carlo (MC) methods for value learning exhibit high variance, particularly for long trajectories, limiting their practicality.",
          "Prior approaches leveraging the triangle inequality in goal-conditioned RL for value learning have not successfully scaled to complex, high-dimensional robotic environments without additional planning."
        ],
        "solution": [
          "Develops Transitive RL (TRL), an algorithm leveraging a divide-and-conquer approach to value learning, specifically designed for offline goal-conditioned reinforcement learning.",
          "Employs a transitive Bellman update rule that uses soft expectile regression to approximate maximization, and critically restricts intermediate subgoals to those observed within the training trajectories.",
          "Utilizes distance-based re-weighting to stabilize training by prioritizing the learning of values for shorter trajectory segments."
        ],
        "keyInsights": [
          "A divide-and-conquer approach to value learning can fundamentally address the \"curse of horizon\" by reducing recursive dependencies from O(T) to O(log T) steps, combining benefits of TD and MC methods.",
          "Preventing value overestimation is critical for scaling transitive value updates; this is effectively achieved by restricting intermediate subgoals to only those present within observed dataset trajectories.",
          "The inherent triangle inequality property of goal-conditioned reinforcement learning can be practically leveraged for complex tasks through careful algorithmic design, specifically using expectile regression and in-trajectory subgoals."
        ],
        "results": [
          "TRL demonstrates state-of-the-art performance on demanding long-horizon robotic tasks like `humanoidmaze-giant` and `puzzle-4x6`, outperforming previous TD-based, MC-based, and other GCRL methods.",
          "The algorithm achieves the best average performance across 10 standard OGBench environments and 50 evaluation tasks, showing strong general competitive capability.",
          "Key components like expectile regression (with \n\n$\n\\kappa > 0.5\n$\n\n), restricting subgoals to in-trajectory states, and distance-based re-weighting were experimentally validated as crucial for TRL's stable and high performance."
        ]
      },
      "image_url": "image/2510.22512v1.png",
      "universal_paper_id": "2510.22512",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 63,
          "last_7_days": 63
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-26T03:32:31.000Z",
      "publication_date": "2025-10-26T03:32:31.000Z",
      "updated_at": "2025-10-29T17:43:22.388Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "deep-reinforcement-learning",
        "optimization-methods",
        "reinforcement-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 12,
      "github_url": "https://github.com/aoberai/trl",
      "distance": 1
    },
    {
      "id": "019a2910-1cbe-74ec-aec4-2772a8f9483c",
      "paper_group_id": "019a2910-1cbe-74ec-aec4-2772a8f9483c",
      "title": "Knocking-Heads Attention",
      "abstract": "多头注意力（MHA）已成为现代大型语言模型的基石，通过并行注意力头增强了表示能力。然而，增加头的数量本质上会削弱单个头的能力，而现有的注意力机制 - 无论是标准的MHA还是其变种如分组查询注意力（GQA）和分组绑定注意力（GTA） - 只是简单地连接孤立头的输出，而没有强有力的互动。为了解决这个限制，我们提出了一种“敲头注意力”（KHA），它使注意力头可以相互“敲击” - 在缩放的点积注意力之前促进跨头特征级别的互动。这是通过在所有头之间应用一个共享的对角初始化的投影矩阵来实现的。对角初始化在训练开始时保留了头特有的专业化，同时允许模型逐渐学习集成的跨头表示。KHA 仅增加了最少的参数和浮点运算，并且可以无缝地集成到 MHA、GQA、GTA 和其他注意力变种中。我们通过在 1T 高质量 token 上训练一个 61 亿参数的 MoE 模型（激活参数为 10.1 亿）来验证 KHA。与基线注意力机制相比，KHA 带来了更优越和更稳定的训练动态，在下游任务中实现了更好的性能。",
      "paper_summary": {
        "summary": "Knocking-Heads Attention (KHA) introduces shared, diagonally-initialized projection matrices to enable efficient cross-head feature-level interactions within multi-head attention mechanisms for large language models. This architectural enhancement significantly reduces loss spikes during pre-training and improves downstream task performance by an average of 1.26 points across various benchmarks, with minimal computational overhead.",
        "originalProblem": [
          "Standard Multi-Head Attention (MHA) operates with isolated heads, limiting expressive power and potentially creating redundancy.",
          "Existing methods for inter-head interaction often introduce high computational overhead or are incompatible with efficient attention implementations like FlashAttention.",
          "Large language model pre-training frequently encounters loss spikes, leading to training instability and wasted computational resources."
        ],
        "solution": [
          "Knocking-Heads Attention (KHA) integrates shared, diagonally-initialized projection matrices into the Q, K, and V paths for feature-level inter-head communication.",
          "KHA offers KHA-Linear (zero inference overhead via matrix absorption) and KHA-MLP (non-linear interaction with comparable parameter count) variants.",
          "A diagonal initialization strategy ensures initial head specialization is preserved, allowing for adaptive learning of cross-head collaboration."
        ],
        "keyInsights": [
          "Applying knocking-heads projections to the value (V) representations yields the most significant performance improvements among Q, K, and V.",
          "Diagonal initialization of the shared projection matrices is critical for maintaining head specialization during early training and achieving stable convergence.",
          "KHA effectively regularizes the training process, leading to a substantial reduction in the frequency and severity of loss spikes."
        ],
        "results": [
          "A 6.1B parameter MoE model equipped with KHA achieved an average 1.26-point improvement across various downstream tasks, including a 4.32-point gain in Language Understanding.",
          "KHA reduced the frequency and severity of loss spikes during 1T token pre-training, leading to a consistently lower training loss (approximately 0.015 points) in later stages.",
          "The method demonstrated broad compatibility, enhancing MHA, GQA, MQA, and GTA across models ranging from 0.44B to 14.6B parameters with minimal computational overhead."
        ]
      },
      "image_url": "image/2510.23052v1.png",
      "universal_paper_id": "2510.23052",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 91,
          "last_7_days": 91
        },
        "public_total_votes": 16
      },
      "first_publication_date": "2025-10-27T06:28:58.000Z",
      "publication_date": "2025-10-27T06:28:58.000Z",
      "updated_at": "2025-10-28T04:25:13.150Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.CL",
        "efficient-transformers",
        "generative-models",
        "parameter-efficient-training",
        "representation-learning",
        "sequence-modeling",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Ant Group",
          "image": null
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
          "name": "Renmin University of China",
          "image": "images/organizations/renmin.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2a0f-9f19-7ded-a883-dcc175b197fc",
      "paper_group_id": "019a2a0f-9f19-7ded-a883-dcc175b197fc",
      "title": "Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts",
      "abstract": "最近在强化学习（RL）方面的进展大幅提升了大规模语言模型的训练，带来了生成质量和推理能力的显著提高。然而，现有研究大多集中于密集模型，而对混合专家（MoE）架构的RL训练仍然探索不足。为了应对MoE训练中常见的不稳定性，我们提出了一种新颖的路由感知方法，旨在优化离策略RL中的重要性采样（IS）权重。具体而言，我们设计了一种由路由逻辑指导的重新缩放策略，有效降低了梯度方差并减轻了训练发散。实验结果表明，我们的方法显著提高了MoE模型的收敛稳定性和最终性能，突显了针对MoE架构的RL算法创新的潜力，并为大规模专家模型的高效训练提供了有前景的方向。",
      "paper_summary": {
        "summary": "A new reinforcement learning algorithm, Router-Shift Policy Optimization (RSPO), enables stable and effective training of Mixture-of-Experts (MoE) models, particularly for mathematical reasoning in large language models. The method achieves an average Pass@1 score of 77.1 on five reasoning benchmarks, surpassing prior baselines and exhibiting robust training convergence.",
        "originalProblem": [
          "Applying reinforcement learning (RL) to Mixture-of-Experts (MoE) models for large language models (LLMs) suffers from training instability.",
          "Dynamic 'router fluctuation' in MoE architectures causes volatility in importance sampling (IS) ratios and can lead to training divergence.",
          "The mismatch between token-level IS ratios and sequence-level rewards is amplified in MoE settings, further contributing to training instability."
        ],
        "solution": [
          "Introduces Router-Shift Policy Optimization (RSPO), an off-policy RL algorithm that incorporates a 'router shift ratio' ($\\gamma_{i,t}$) to adaptively reweight importance sampling ratios.",
          "The $\\gamma_{i,t}$ term quantifies the deviation in expert routing decisions between old and current policies, down-weighting updates for tokens with larger shifts; gradient flow through $\\gamma_{i,t}$ is explicitly stopped to maintain stability.",
          "Utilizes sequence-level importance ratios with token-level clipping to mitigate variance mismatch while preserving granular control over token updates."
        ],
        "keyInsights": [
          "Explicitly reweighting policy updates based on router shifts (router-aware importance sampling) is crucial for stabilizing RL training in MoE architectures.",
          "Disabling gradient flow through the router shift ratio itself is essential to prevent early training collapse and ensure stable optimization.",
          "Rigid router stabilization strategies like freezing the router or simple replay mechanisms are insufficient and can negatively impact model adaptability and performance."
        ],
        "results": [
          "RSPO achieved an average Pass@1 score of 77.1 on mathematical reasoning benchmarks (AIME24, AMC23, MATH500, Minerva, OlympiadBench) using a Qwen3-30B-A3B model, outperforming GRPO (71.5), GSPO (76.4), and GMPO (76.4).",
          "Demonstrated superior training stability, exhibiting consistent high validation scores throughout training, unlike GRPO which showed pronounced performance collapse.",
          "Ablation studies confirmed that applying `stop_grad` to the router shift ratio is critical for stable training, as allowing gradients through it led to early training collapse."
        ]
      },
      "image_url": "image/2510.23027v1.png",
      "universal_paper_id": "2510.23027",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 89,
          "last_7_days": 89
        },
        "public_total_votes": 13
      },
      "first_publication_date": "2025-10-27T05:47:48.000Z",
      "publication_date": "2025-10-27T05:47:48.000Z",
      "updated_at": "2025-10-28T09:04:18.201Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "cs.LG",
        "deep-reinforcement-learning",
        "generative-models",
        "optimization-methods",
        "parameter-efficient-training",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Peking University",
          "image": "images/organizations/peking.png"
        },
        {
          "name": "Microsoft",
          "image": "images/organizations/microsoft.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a30aa-2711-7081-ac61-b28b48b9f2fa",
      "paper_group_id": "019a30aa-2711-7081-ac61-b28b48b9f2fa",
      "title": "HRM-Agent: Training a recurrent reasoning model in dynamic environments using reinforcement learning",
      "abstract": "层次推理模型（HRM）在其小巧体积下展现了令人印象深刻的推理能力，但仅被应用于监督的、静态的、完全可观察的问题。HRM的一个优势是其能够根据问题的难度调整计算工作量。然而，在当前形式下，它无法在问题是动态的、不确定的或部分可观察的情况下整合和重用先前时间步的计算，或者在正确行动未定义的情况下应用，这些都是许多现实世界问题的特征。\n本文介绍了HRM-Agent，一个仅通过强化学习进行训练的HRM变体。我们展示了HRM能够在动态和不确定的迷宫环境中学习导航到目标。近期的研究表明，HRM的推理能力源于其递归推理过程。我们探讨了递归推理过程的动态性，并发现有证据表明它成功地重用了来自早期环境时间步的计算。",
      "paper_summary": {
        "summary": "Melbourne, Australia, and Cerenaut AI researchers adapted the Hierarchical Reasoning Model (HRM) for reinforcement learning in dynamic maze environments, demonstrating that its recurrent internal state can maintain and adapt learned plans across environmental changes. The HRM-Agent achieved approximately 99% success rates in navigation tasks while efficiently reusing prior computations through its carried-forward latent state.",
        "originalProblem": [
          "Existing reasoning models like Large Language Models (LLMs) are often inefficient and struggle with adaptive, sustained reasoning in dynamic and partially observable environments.",
          "The Hierarchical Reasoning Model (HRM) was previously limited to supervised, static, and fully-observable problems, restricting its real-world applicability.",
          "Reinforcement learning agents require mechanisms to effectively handle hierarchical, long-horizon planning and maintain consistent internal states across dynamic environmental changes."
        ],
        "solution": [
          "Adapted the Hierarchical Reasoning Model (HRM) by replacing its output head with a Deep Q-Network (DQN) head for action value prediction.",
          "Trained the HRM-Agent entirely from scratch using reinforcement learning, without any supervised losses or human-defined problem decompositions.",
          "Introduced a \"Carry Z\" mechanism where the agent's recurrent latent state from the previous environment step is carried forward to the next, promoting computational reuse."
        ],
        "keyInsights": [
          "The intrinsic recurrent reasoning capabilities of HRM can be successfully leveraged within an RL framework to develop adaptive agents for dynamic and uncertain environments.",
          "Carrying forward the recurrent latent state (z) allows the agent to efficiently reuse computations, accelerate internal plan convergence, and maintain consistency in its reasoning across environmental timesteps.",
          "The agent's recurrent state appears to function as an internal \"plan\" or \"belief state\" that dynamically adapts to environmental changes without being reset."
        ],
        "results": [
          "The HRM-Agent achieved a high success rate of approximately 99% in goal navigation within both dynamic Four-Rooms and random maze environments.",
          "Demonstrated efficient path planning, with mean episode lengths approaching theoretical optimums, indicating learned deliberate reasoning.",
          "The \"Carry Z\" variant consistently led to faster convergence of the recurrent latent state and exhibited greater consistency in internal plans compared to the \"Reset Z\" variant."
        ]
      },
      "image_url": "image/2510.22832v1.png",
      "universal_paper_id": "2510.22832",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 33,
          "last_7_days": 33
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-26T21:01:04.000Z",
      "publication_date": "2025-10-26T21:01:04.000Z",
      "updated_at": "2025-10-29T15:50:48.849Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "online-learning",
        "reasoning",
        "reinforcement-learning",
        "sequence-modeling",
        "Statistics",
        "stat.ML"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a29c7-fe4c-7929-8134-929a2bda71a9",
      "paper_group_id": "019a29c7-fe4c-7929-8134-929a2bda71a9",
      "title": "FARMER: Flow AutoRegressive Transformer over Pixels",
      "abstract": "直接建模原始数据分布的显式似然性是机器学习领域的一个关键主题，通过自回归建模实现了大型语言模型的扩展成功。然而，针对视觉像素数据的连续自回归建模在处理极长序列和高维空间时面临困难。本文提出了FARMER，这是一种新颖的端到端生成框架，将归一化流（NF）和自回归（AR）模型统一起来，以便直接从原始像素进行可处理的似然估计和高质量图像合成。FARMER采用可逆自回归流将图像转换为潜在序列，其分布通过自回归模型隐式建模。为了解决像素级建模中的冗余和复杂性，我们提出了一种自监督维度降低方案，将NF潜在通道划分为信息性和冗余组，从而实现更有效率的AR建模。此外，我们设计了一种一步蒸馏方案，以显著加快推理速度，并引入了一种基于重采样的无分类器引导算法，以提升图像生成质量。大量实验证明，FARMER在提供精确似然性和可扩展训练的同时，与现有像素基础生成模型相比，表现出竞争力的性能。",
      "paper_summary": {
        "summary": "A generative framework named FARMER unifies invertible Autoregressive Flows with Autoregressive Transformers to directly model raw image pixels, providing explicit likelihood estimation. Developed by researchers from ByteDance and academic institutions, the model achieves a FID of 3.60 on ImageNet 256x256, outperforming JetFormer by 3.04 FID points, and accelerates inference by nearly 4x through a one-step distillation scheme.",
        "originalProblem": [
          "Existing state-of-the-art generative models for images (GANs, VAEs, diffusion models) often lack tractable, explicit likelihood estimation, essential for tasks like anomaly detection and principled model comparison.",
          "Direct application of continuous autoregressive models to high-dimensional image pixels is computationally expensive and struggles with long-range dependencies, while Normalizing Flows often degrade image quality by forcing complex distributions onto simple priors.",
          "Autoregressive Flows, despite their expressiveness, suffer from inherently slow sequential reverse inference, which hinders their practical applicability."
        ],
        "solution": [
          "An end-to-end framework directly models raw image pixels by unifying an invertible Autoregressive Flow (AF) to transform images into a latent sequence, whose distribution is then modeled by an Autoregressive Transformer using Gaussian Mixture Models.",
          "The framework incorporates a self-supervised dimension reduction method that partitions latent channels into informative and redundant parts, and utilizes a novel resampling-based Classifier-Free Guidance for improved generation quality.",
          "To overcome slow sequential inference, a one-step distillation technique trains a student AF to perform the entire reverse transformation in a single, parallel step, significantly accelerating the process."
        ],
        "keyInsights": [
          "A self-supervised dimension reduction technique effectively disentangles structural and fine-grained information in the latent space, critically improving image quality by managing redundancy in pixel-level autoregressive modeling.",
          "Resampling-based Classifier-Free Guidance is essential for achieving high-fidelity image generation, significantly enhancing performance compared to naive guidance within the Autoregressive Flow framework.",
          "One-step distillation dramatically accelerates the inherently slow sequential reverse inference of Autoregressive Flows, making them practical for real-world applications with minimal impact on generation quality."
        ],
        "results": [
          "The FARMER 1.9B model achieved a Fréchet Inception Distance (FID) of 3.60 on ImageNet 256x256, outperforming the most comparable baseline, JetFormer, by a margin of 3.04 FID points.",
          "The framework effectively synthesizes diverse, high-quality images, preserving fine-grained details that are often blurred by latent compression methods in other generative models.",
          "Inference time per image was accelerated by nearly 4x, reducing from 0.2189 seconds to 0.0567 seconds, through one-step distillation, while maintaining comparable image quality (FID of 5.63 post-distillation versus 5.55 for the original)."
        ]
      },
      "image_url": "image/2510.23588v1.png",
      "universal_paper_id": "2510.23588",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 107,
          "last_7_days": 107
        },
        "public_total_votes": 14
      },
      "first_publication_date": "2025-10-27T17:54:08.000Z",
      "publication_date": "2025-10-27T17:54:08.000Z",
      "updated_at": "2025-10-28T07:46:03.981Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "knowledge-distillation",
        "representation-learning",
        "self-supervised-learning",
        "sequence-modeling",
        "synthetic-data",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "NUS",
          "image": null
        },
        {
          "name": "USTC",
          "image": null
        },
        {
          "name": "ANU",
          "image": null
        },
        {
          "name": "ByteDance Seed China",
          "image": null
        },
        {
          "name": "ByteDance Seed Singapore",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2e02-ce20-7df9-8ca9-ef60ea5853bb",
      "paper_group_id": "019a2e02-ce20-7df9-8ca9-ef60ea5853bb",
      "title": "Critique-RL: Training Language Models for Critiquing through Two-Stage Reinforcement Learning",
      "abstract": "训练批评语言模型以评估和提供模型输出反馈是改善大型语言模型在复杂推理任务中的一种有前景的方法。然而，现有方法通常依赖于更强的监督者来标注批评数据。为了解决这一问题，我们提出了Critique-RL，一种在线强化学习方法，用于在没有较强监督的情况下开发批评语言模型。我们的方法采用双玩家范式：演员生成响应，评论者提供反馈，演员相应地完善响应。我们首先揭示，仅依赖演员输出的间接奖励信号进行强化学习优化通常导致不理想的评论者：尽管他们的有用性（即提供建设性反馈）有所改善，但区分能力（即判断响应是否高质量的能力）仍然较差，导致性能提升有限。为了解决这个问题，Critique-RL采用了两阶段优化策略。在第一阶段，它使用基于规则的直接奖励信号强化评论者的区分能力；在第二阶段，它引入基于演员优化的间接奖励，以提高评论者的有用性，同时通过适当的正则化维持其区分能力。针对各种任务和模型的大量实验表明，Critique-RL在性能上显著提升。例如，Qwen2.5-7B在领域内任务上获得了9.02%的提升，在领域外任务上获得了5.70%的提升，突显了其潜力。",
      "paper_summary": {
        "summary": "Critique-RL, developed by Fudan University and ByteDance Seed, introduces a two-stage reinforcement learning framework to train language models for critiquing other LLM outputs without strong supervision. This method jointly optimizes a critic's discriminability and helpfulness, leading to actor models achieving up to a 9.02% accuracy gain on in-domain tasks and 5.70% on out-of-domain tasks.",
        "originalProblem": [
          "Existing methods for training LLM critique models heavily depend on expensive, hard-to-scale human or oracle supervision for critique data annotation.",
          "Prior reinforcement learning approaches for training critics using indirect reward signals often fail to jointly optimize both the critic's discriminability (accuracy in judging correctness) and helpfulness (quality of feedback).",
          "Many prompt engineering methods for critiquing implicitly assume an external oracle verifier during testing, failing to address the critical need for a critic to autonomously determine output correctness."
        ],
        "solution": [
          "Critique-RL employs a two-player actor-critic reinforcement learning framework where a critic model assesses an actor's response and guides its refinement.",
          "A novel two-stage RL strategy is introduced: Stage I explicitly optimizes the critic's discriminability using direct, rule-based reward signals, and Stage II then enhances helpfulness using indirect rewards while maintaining discriminability with regularization terms.",
          "The framework uses online RL to train the critic without stronger supervision and without requiring an oracle verifier at test time, addressing limitations of previous methods."
        ],
        "keyInsights": [
          "Indirect reward signals, commonly used in prior RL-based critic training, are insufficient to jointly optimize a critic's discriminability and helpfulness, often leading to critics that are either overly conservative or aggressive.",
          "Explicitly disentangling and sequentially optimizing discriminability before helpfulness, with appropriate regularization, is crucial for developing robust and effective critique models.",
          "An autonomous critique-refinement loop trained through this two-stage RL can significantly improve actor LLM performance while also being more compute-efficient than parallel sampling methods."
        ],
        "results": [
          "Critique-RL achieved up to a 9.02% average accuracy gain on in-domain reasoning tasks and 5.70% on out-of-domain tasks for Qwen2.5-7B actor models.",
          "The method significantly outperformed other online RL baselines, showing an average improvement of 5.11 points in actor accuracy and 12.69 points in critic discriminability over Retroformer for Qwen2.5-7B.",
          "Ablation studies confirmed the necessity of both RL stages and the direct discriminability reward, demonstrating that the two-stage design is critical for achieving high performance in both discriminability and helpfulness."
        ]
      },
      "image_url": "image/2510.24320v1.png",
      "universal_paper_id": "2510.24320",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 47,
          "last_7_days": 47
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-28T11:37:01.000Z",
      "publication_date": "2025-10-28T11:37:01.000Z",
      "updated_at": "2025-10-29T03:28:47.136Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "explainable-ai",
        "fine-tuning",
        "reasoning",
        "reinforcement-learning",
        "transformers",
        "weak-supervision"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/WooooDyy/Critique-RL",
      "distance": 1
    },
    {
      "id": "019a2ef4-7c24-718a-b604-4524ec8eb0ed",
      "paper_group_id": "019a2ef4-7c24-718a-b604-4524ec8eb0ed",
      "title": "Group Relative Attention Guidance for Image Editing",
      "abstract": "最近，基于扩散变换器模型的图像编辑迅速发展。然而，现有的编辑方法往往缺乏对编辑程度的有效控制，限制了其实现更定制化结果的能力。为了解决这一限制，我们研究了DiT模型中的MM-注意力机制，并观察到查询和关键令牌共享一个仅依赖于层的偏置向量。我们将这个偏置解释为代表模型固有的编辑行为，而每个令牌与其对应偏置之间的增量编码了内容特定的编辑信号。基于这一洞察，我们提出了组相对注意力引导，这是一种简单但有效的方法，可以重加权不同令牌的增量值，以调节模型相对于编辑指令对输入图像的关注，使得在不进行任何调优的情况下实现持续和细粒度的编辑强度控制。在现有图像编辑框架上进行的大量实验表明，GRAG可以用少至四行代码进行集成，并且始终提高编辑质量。此外，与常用的无分类器引导相比，GRAG在编辑程度的控制上实现了更平滑和更精确的效果。我们的代码将会在这个网址发布。",
      "paper_summary": {
        "summary": "Researchers from Tianjin University and Kuaishou Technology introduced Group Relative Attention Guidance (GRAG), a plug-and-play mechanism for Diffusion Transformers that enables continuous and fine-grained control over editing intensity in text-driven image editing. GRAG leverages a newly identified bias vector within multi-modal attention layers to achieve a precise balance between source image fidelity and editing instruction adherence.",
        "originalProblem": [
          "Lack of fine-grained control over editing strength in text-driven image editing models.",
          "Users struggle to balance source image fidelity with adherence to editing instructions.",
          "Inefficiency due to reliance on tedious prompt engineering or multiple inference attempts for desired editing outcomes."
        ],
        "solution": [
          "Proposes Group Relative Attention Guidance (GRAG), a lightweight, plug-and-play guidance mechanism for Diffusion Transformers (DiTs).",
          "GRAG modulates the Multi-Modal Attention (MM-Attention) mechanism by reweighting content-specific delta values from a bias vector.",
          "It uses a tunable parameter, \"δ\", to control the continuous intensity of the edit, specifically scaling the token-specific deviation component."
        ],
        "keyInsights": [
          "Multi-Modal Attention (MM-Attention) in DiT models contains a shared, layer-dependent bias vector representing the model's inherent editing behavior.",
          "Deviations from this bias vector (deltas) encode content-specific editing signals.",
          "Modulating these content-specific delta values allows for precise control over the extent to which conditional signals (like edit instructions) influence the output."
        ],
        "results": [
          "GRAG enables continuous and fine-grained control over editing strength, allowing smooth transitions between minimal and strong edits.",
          "Quantitatively improved content preservation (lower LPIPS, higher SSIM) and overall EditScore for training-based models like Step1X-Edit and Qwen-Edit.",
          "Ablation studies showed that the \"δ\" parameter alone provides the most effective and continuous control, outperforming Classifier-Free Guidance and other parameter combinations."
        ]
      },
      "image_url": "image/2510.24657v1.png",
      "universal_paper_id": "2510.24657",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 45,
          "last_7_days": 45
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-28T17:22:44.000Z",
      "publication_date": "2025-10-28T17:22:44.000Z",
      "updated_at": "2025-10-29T07:52:45.860Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.CV",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "model-interpretation",
        "multi-modal-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Tianjin University",
          "image": null
        },
        {
          "name": "Kuaishou Technology",
          "image": null
        },
        {
          "name": "Kolors Team, Kuaishou Technology",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 6,
      "github_url": "https://github.com/little-misfit/GRAG-Image-Editing",
      "distance": 1
    },
    {
      "id": "019a2db1-808e-7e30-87bc-cf62ab73dd1b",
      "paper_group_id": "019a2db1-808e-7e30-87bc-cf62ab73dd1b",
      "title": "SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity",
      "abstract": "近期文本到语音（TTS）合成的进展显著提高了语音的表现力和自然性。然而，大多数现有系统针对单一说话者合成进行了优化，无法生成连贯的多说话者对话语音。本技术报告介绍了SoulX-Podcast，这是一个旨在进行播客风格的多轮、多说话者对话语音生成的系统，同时在传统TTS任务中也达到了先进的性能。\n\n为了满足多轮口语对话对自然性的更高要求，SoulX-Podcast整合了一系列语用控制，并支持普通话和英语，以及多种中文方言，包括四川话、河南话和粤语，从而实现更个性化的播客风格语音生成。实验结果表明，SoulX-Podcast能够持续生成超过90分钟的对话，语音音色稳定，说话者之间的切换流畅。此外，发声者展示了上下文适应性的韵律，随着对话的发展反映出自然的节奏和语调变化。在多个评估指标上，SoulX-Podcast在单人独白TTS和多轮对话语音合成方面均达到了先进的性能。",
      "paper_summary": {
        "summary": "SoulX-Podcast introduces an LLM-driven generative framework for creating realistic, long-form, multi-speaker podcasts, incorporating diverse Chinese dialects and controllable paralinguistic cues. The system achieves state-of-the-art performance in multi-turn dialogue synthesis, exhibiting the lowest Character Error Rate (2.20) and highest cross-speaker consistency (0.599) on the Chinese ZipVoice-Dia benchmark, alongside strong zero-shot monologue capabilities.",
        "originalProblem": [
          "Current state-of-the-art Text-to-Speech (TTS) models are primarily optimized for single-speaker, monologue-style generation, failing to handle multi-speaker conversational complexities.",
          "Existing dialogue TTS systems lack fine-grained control over paralinguistic cues (e.g., laughter, sighs) and offer limited support for linguistic diversity beyond standard languages, such as regional dialects.",
          "Maintaining long-form coherence, stable speaker identities, and smooth transitions in continuous, extended conversational speech generation poses a significant challenge."
        ],
        "solution": [
          "An LLM-driven two-stage generative framework, based on Qwen3-1.7B, is developed, extending its text codebook with speech, paralinguistic, and dialectal tokens.",
          "A comprehensive data processing and annotation pipeline creates a massive 1.3 million-hour corpus, meticulously segmented, diarized, quality-filtered, and enriched with paralinguistic and dialectal labels.",
          "A text-speech interleaved token organization facilitates multi-turn, multi-speaker dialogue, combined with a curriculum learning strategy and context regularization for long-form coherence.",
          "A novel Dialect-Guided Prompting (DGP) strategy enables cross-dialectal zero-shot voice cloning by prepending dialect-typical sentences to input text during inference."
        ],
        "keyInsights": [
          "Explicitly integrating paralinguistic and dialectal tokens within an LLM's extended text codebook, alongside a text-speech interleaved sequence, effectively enables diverse and expressive conversational synthesis.",
          "Rigorous data processing pipelines, encompassing vocal separation, diarization, dual-ASR transcription, and speaker purity refinement, are fundamental for training robust multi-speaker dialogue TTS systems.",
          "A curriculum learning strategy, enhanced with context regularization, promotes long-term coherence and stable speaker identities in multi-turn dialogue generation by progressively emphasizing semantic over acoustic context.",
          "Cross-dialectal voice cloning can be achieved through Dialect-Guided Prompting (DGP), which uses a short, dialect-typical sentence as a prompt to steer synthesis towards a target dialect."
        ],
        "results": [
          "SoulX-Podcast achieved the lowest Character Error Rate (CER) of 2.20 for Chinese and Word Error Rate (WER) of 2.27 for English dialogue on the ZipVoice-Dia test set, with the highest cross-speaker consistency (0.599 for Chinese).",
          "In zero-shot monologue generation, the system recorded the lowest Chinese CER of 1.10 on Seed-TTS-eval, confirming high intelligibility, and maintained competitive speaker similarity scores.",
          "The system demonstrated effective control over five paralinguistic events, achieving an overall recognition accuracy of 0.82, including 1.00 for laughter and 0.85 for sighs.",
          "Cross-dialectal voice cloning was successfully implemented for Sichuanese, Henanese, and Cantonese using Dialect-Guided Prompting, maintaining consistent speaker similarity across dialects."
        ]
      },
      "image_url": "image/2510.23541v2.png",
      "universal_paper_id": "2510.23541",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 58,
          "last_7_days": 58
        },
        "public_total_votes": 8
      },
      "first_publication_date": "2025-10-27T17:15:05.000Z",
      "publication_date": "2025-10-28T17:23:22.000Z",
      "updated_at": "2025-10-29T01:59:58.862Z",
      "topics": [
        "Computer Science",
        "cs.SD",
        "eess.AS",
        "Electrical Engineering and Systems Science"
      ],
      "organization_info": [
        {
          "name": "Northwestern Polytechnical University",
          "image": null
        },
        {
          "name": "Shanghai Jiao Tong University",
          "image": "images/organizations/shanghai-jiao-tong-university.png"
        },
        {
          "name": "Soul AI Lab",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    }
  ],
  "page": 0
};