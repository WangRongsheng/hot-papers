const papersData = {
  "papers": [
    {
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们提出了Tongyi DeepResearch，这是一个代理型大型语言模型，专门设计用于长期、深入的信息探索研究任务。为了激励自主深度研究能力，Tongyi DeepResearch是通过一种端到端的训练框架开发的，该框架结合了代理中期训练和代理后期训练，使得在复杂任务中能够进行可扩展的推理和信息寻求。我们设计了一种高度可扩展的数据合成流程，完全自动化，无需依赖昂贵的人类标注，支持所有训练阶段。通过为每个阶段构建定制的环境，我们的系统实现了整个过程中的稳定和一致的交互。Tongyi DeepResearch拥有305亿个参数，其中每个token仅激活33亿个参数，在一系列代理深度研究基准测试中实现了最先进的性能，包括人类的最后考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES和xbench-DeepSearch-2510。我们开源了模型、框架和完整解决方案，以赋能社区。",
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
        "total_votes": 31,
        "visits_count": {
          "all": 1032,
          "last_7_days": 1032
        },
        "public_total_votes": 82
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
      "id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "paper_group_id": "019a330b-5931-7217-bd0d-53a1f43c09bd",
      "title": "Scaling Latent Reasoning via Looped Language Models",
      "abstract": "现代的大型语言模型（LLMs）主要通过显式文本生成进行“思考”，如思维链（CoT），这将推理推迟到训练后，并未充分利用预训练数据。我们提出并开源了Ouro，名为递归的乌罗波罗斯（Ouroboros），这是一个预训练的循环语言模型（LoopLM）系列，通过（i）在潜在空间中的迭代计算，（ii）用于学习深度分配的熵正则化目标，和（iii）扩展到7.7T个标记来构建推理于预训练阶段。Ouro 1.4B和2.6B模型在广泛的基准测试中表现优越，与高达12B的最先进LLMs的结果相匹配。通过控制实验，我们表明这一优势并非来自知识容量的增加，而是来自更优越的知识操控能力。我们还显示，LoopLM产生的推理痕迹与最终输出的对齐程度优于显式CoT。我们希望我们的结果展示了LoopLM作为推理时代新型扩展方向的潜力。我们的模型可以在此找到：这个HTTP URL。",
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
        "total_votes": 12,
        "visits_count": {
          "all": 419,
          "last_7_days": 419
        },
        "public_total_votes": 34
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
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLM）的推理能力方面展现出了显著的潜力。然而，RL在LLM上的成功严重依赖于人类精心策划的数据集和可验证的奖励，这限制了其可扩展性和通用性。最近受到游戏和围棋成功启发的自我对弈强化学习方法，旨在增强LLM的推理能力，而无需人工标注的数据。然而，这些方法主要依赖一个有反馈的基础环境（例如，Python解释器或游戏引擎）；将其扩展到一般领域仍然具有挑战性。为了解决这些问题，我们提出了多智能体进化（MAE）框架，该框架使LLM能够自我进化以解决各种任务，包括数学、推理和常识问答。MAE的核心设计基于一个由三个相互作用的智能体（提问者、求解者、评审者）组成的三元组，这些智能体源自同一个LLM，并应用强化学习来优化它们的行为。提问者生成问题，求解者尝试解决方案，而评审者在共同进化的过程中评估二者。Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试中实现了平均4.54%的提升。这些结果突显了MAE作为一种可扩展且数据高效的方法，通过最小依赖人类策划的监督来增强LLM的通用推理能力。",
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
        "total_votes": 14,
        "visits_count": {
          "all": 567,
          "last_7_days": 567
        },
        "public_total_votes": 54
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
      "id": "019a315b-20e8-7941-a237-262a8aeeb18c",
      "paper_group_id": "019a315b-20e8-7941-a237-262a8aeeb18c",
      "title": "An efficient probabilistic hardware architecture for diffusion-like models",
      "abstract": "概率人工智能的扩散促进了对专用随机计算机的提议。尽管这些提议在效率上有望带来改善，但由于它们依赖于根本有限的建模技术和奇特的、不可扩展的硬件，因此未能获得广泛认可。在本研究中，我们通过提出一种全晶体管概率计算机来解决这些不足，该计算机在硬件层面实现强大的去噪模型。系统级分析表明，基于我们架构的设备在一个简单的图像基准测试中可以实现与GPU相当的性能，并且能耗大约低10,000倍。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 232,
          "last_7_days": 232
        },
        "public_total_votes": 22
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
      "id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "paper_group_id": "019a306d-221a-7ffd-8c54-65a431dadafb",
      "title": "Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents",
      "abstract": "关于大规模监督微调AI代理的公共研究结果仍然相对稀少，因为收集代理训练数据面临独特的挑战。在这项工作中，我们认为瓶颈并不是缺乏基础数据源，而是各种数据在异构格式、工具和接口之间高度碎片化。为此，我们提出了代理数据协议（ADP），这是一种轻量级的表示语言，充当不同格式的代理数据集与统一的下游代理训练管道之间的“中介语”。ADP的设计足够表达多种任务，包括API/工具使用、浏览、编码、软件工程和一般代理工作流程，同时保持解析和训练的简单性，而无需在每个数据集级别进行工程化。在实验中，我们将13个现有的代理训练数据集统一为ADP格式，并将标准化的ADP数据转换为多个代理框架的训练准备格式。我们在这些数据上进行了监督微调，平均表现提升约20%，在标准编码、浏览、工具使用和研究基准上实现了最先进或者接近最先进的性能，而无需进行领域特定的调优。所有代码和数据都已公开发布，希望ADP能够帮助降低标准化、可扩展和可重复的代理训练的门槛。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 234,
          "last_7_days": 234
        },
        "public_total_votes": 25
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
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于LLM的网页代理在信息检索方面展现了巨大的潜力，但在长时程任务中的有效性受到背景管理的基本权衡影响。当前的ReAct基础代理由于积累了嘈杂的原始历史，面临背景饱和的问题，而每一步固定总结完整历史的方法则有可能导致关键信息的不可逆损失。为了解决这些问题，我们推出了AgentFold，这是一种以主动背景管理为中心的新型代理范式，灵感来源于人类认知过程中的回顾整合。AgentFold将其背景视为一个动态的认知工作空间，积极进行塑造，而不是一个被动的记录。每一步，它学习执行一个“折叠”操作，在多个尺度上管理其历史轨迹：它可以进行细粒度的凝聚以保留重要的细节，也可以进行深度的整合以抽象出整个多步子任务。在显著的基准测试中，结果令人瞩目：通过简单的监督微调（不需要持续预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上取得了36.2%、在BrowseComp-ZH上取得了47.3%的成绩。值得注意的是，这一表现不仅超过或匹配了规模远大于自身的开源模型，如DeepSeek-V3.1-671B-A37B，还超过了诸如OpenAI的o4-mini等领先的专有代理。",
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
          "all": 208,
          "last_7_days": 208
        },
        "public_total_votes": 27
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
      "id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "paper_group_id": "019a38b6-4852-748b-80ff-8fca8b517243",
      "title": "Defeating the Training-Inference Mismatch via FP16",
      "abstract": "强化学习（RL）对大型语言模型（LLM）的微调常常因训练政策和推断政策之间的数值不匹配而导致不稳定性。尽管之前的研究尝试通过算法修正或工程对齐来减轻这一问题，但我们表明其根本原因在于浮点精度本身。尽管广泛采用的BF16具有较大的动态范围，但它引入了大量的舍入错误，破坏了训练和推断之间的一致性。在本研究中，我们证明简单地回退到\\textbf{FP16}可以有效消除这种不匹配。这一变化简单，现代框架完全支持，只需几行代码的更改，并且不需对模型架构或学习算法进行修改。我们的结果表明，使用FP16在不同任务、算法和框架中通常能够实现更稳定的优化、更快的收敛和更强的性能。我们希望这些发现能激励人们更广泛地重新考虑RL微调中的精度权衡。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 69,
          "last_7_days": 69
        },
        "public_total_votes": 10
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
      "id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "paper_group_id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "title": "Uniform Discrete Diffusion with Metric Path for Video Generation",
      "abstract": "连续空间视频生成快速发展，而离散方法由于误差积累和长时段不一致性滞后。在本研究中，我们重新审视了离散生成建模，并提出了Uniform discRete diffuSion with metric pAth（URSA），这是一个简单而强大的框架，弥合了可扩展视频生成与连续方法之间的差距。URSA的核心是将视频生成任务表述为离散时空令牌的迭代全局优化。它整合了两个关键设计：线性化度量路径和分辨率依赖的时间步长转换机制。这些设计使URSA能够高效地扩展到高分辨率图像合成和长时段视频生成，同时所需的推理步骤显著减少。此外，我们还引入了一种异步时间微调策略，将多种任务统一于单一模型中，包括插值和图像到视频的生成。在具有挑战性的视频和图像生成基准上的广泛实验表明，URSA在性能上始终优于现有的离散方法，并且达到了与最先进的连续扩散方法相当的性能。代码和模型可在该https URL获取。",
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
          "all": 212,
          "last_7_days": 212
        },
        "public_total_votes": 22
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
      "id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "paper_group_id": "019a2dc8-3214-77f5-8ba1-95cbcf79d717",
      "title": "SPICE: Self-Play In Corpus Environments Improves Reasoning",
      "abstract": "自我提升系统需要与环境互动以实现持续的适应。我们引入了SPICE（自我对弈语料环境），这是一种强化学习框架，单一模型承担两个角色：挑战者从大型语料库中挖掘文档以生成多样的推理任务，推理者则负责解决这些任务。通过对抗性动态，挑战者在推理者能力的前沿创建自动课程，而语料的基础则提供了丰富的、几乎用不完的外部信号，必要以支持持续的改进。与现有的无基础自我对弈方法所提供的有限益处不同，SPICE在多个模型类别上的数学（+8.9%）和一般推理（+9.8%）基准上实现了一致的提升。我们的分析揭示了文档基础是SPICE中关键的成分，它能够持续生成自己日益具有挑战性的目标并实现这些目标，从而实现持续的自我提升。",
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
        "total_votes": 7,
        "visits_count": {
          "all": 156,
          "last_7_days": 156
        },
        "public_total_votes": 24
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
      "id": "019a330f-ab28-7eba-9849-ff7b835837c9",
      "paper_group_id": "019a330f-ab28-7eba-9849-ff7b835837c9",
      "title": "The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution",
      "abstract": "现实世界中的语言代理必须处理跨多个应用程序的复杂多步骤工作流程。例如，一个代理可以通过与日历和文件系统协调来管理电子邮件，或者监控生产数据库以检测异常，并根据操作手册生成报告。然而，现有的语言代理基准测试往往集中于狭窄的领域或简化的任务，缺乏评估代理在真实世界表现所需的多样性、真实性和长期复杂性。为了解决这个问题，我们推出了工具十项全能赛（称为Toolathlon），这是一项为语言代理提供多样化应用和工具的基准测试，具有真实的环境设置和可靠的执行基础评估。Toolathlon涵盖32个软件应用和604个工具，从日常平台如Google日历和Notion，到专业工具如WooCommerce、Kubernetes和BigQuery。大多数工具基于我们可能已经修订或自行实施的高质量模型上下文协议（MCP）服务器。与以往的工作不同，后者主要确保功能现实性，但环境状态多样性有限，我们提供真实软件的真实初始环境状态，例如有数十名学生的Canvas课程或实际的财务电子表格。该基准总共包括108个手动获取或定制的任务，要求在完成时平均与多个应用互动约20个回合。每个任务都可以通过专门的评估脚本严格验证。对最先进模型的全面评估突显了它们的显著缺陷：表现最佳的模型Claude-4.5-Sonnet的成功率仅为38.6%，平均召唤工具的回合数为20.2，而顶级开放权重模型DeepSeek-V3.2-Exp的成功率为20.1%。我们期待Toolathlon推动更强大语言代理的发展，以执行现实世界的长期任务。",
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
          "all": 98,
          "last_7_days": 98
        },
        "public_total_votes": 13
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
      "id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "paper_group_id": "019a380e-305a-71ac-bef8-bb045bbbded1",
      "title": "Emu3.5: Native Multimodal Models are World Learners",
      "abstract": "我们推出了Emu3.5，这是一个大规模的多模态世界模型，能够原生地预测视觉和语言的下一状态。Emu3.5在一个包含超过10万亿个标记的视觉-语言交错数据语料库上进行了端到端的预训练，主要源自互联网视频的连续帧和文字记录。该模型自然接受交错的视觉-语言输入，并生成交错的视觉-语言输出。Emu3.5进一步通过大规模强化学习进行后训练，以增强多模态推理和生成。为了提高推理效率，我们提出了离散扩散适应（DiDA），将逐 token 解码转换为双向并行预测，使每张图像的推理速度加快约20倍，而不牺牲性能。Emu3.5展现了强大的原生多模态能力，包括长时间范围的视觉-语言生成、任意到图像（X2I）生成以及复杂的丰富文本图像生成。它还表现出可泛化的世界建模能力，使得在多样场景和任务中能够进行时空一致的世界探索和开放世界的具身操作。作为比较，Emu3.5在图像生成和编辑任务上达到了与Gemini 2.5 Flash Image（Nano Banana）相当的性能，并在一系列交错生成任务上展示了卓越的结果。我们在这个https URL上开源Emu3.5，以支持社区研究。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 59,
          "last_7_days": 59
        },
        "public_total_votes": 10
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
      "id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "paper_group_id": "019a3883-8556-7e55-bfb5-d6dd27a39c0b",
      "title": "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender",
      "abstract": "在推荐系统中，扩展特征交互模块（例如，Wukong、RankMixer）或用户行为序列模块（例如，LONGER）已取得显著成功。然而，这些努力通常在不同的轨道上进行，这不仅阻碍了双向信息交换，还阻止了统一优化和扩展。在本文中，我们提出了OneTrans，一种统一的Transformer骨干网，能够同时执行用户行为序列建模和特征交互。OneTrans采用统一的分词器，将顺序和非顺序属性转换为单个令牌序列。堆叠的OneTrans模块在相似的顺序令牌之间共享参数，同时为非顺序令牌分配特定的参数。通过因果注意力和跨请求KV缓存，OneTrans实现了中间表示的预计算和缓存，在训练和推理过程中显著降低了计算成本。在工业规模数据集上的实验结果表明，OneTrans在参数增加时表现出良好的扩展性，始终优于强基线，并在在线A/B测试中实现了每用户GMV提升5.68%。",
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
          "all": 57,
          "last_7_days": 57
        },
        "public_total_votes": 9
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
      "id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "paper_group_id": "019a3829-2111-73c8-ab76-f934bd1469c0",
      "title": "Kimi Linear: An Expressive, Efficient Attention Architecture",
      "abstract": "我们介绍了Kimi Linear，一种混合线性注意力架构，它首次在多种场景下的公正比较中超越了全注意力，包括短上下文、长上下文和强化学习（RL）扩展范畴。其核心是Kimi Delta Attention（KDA），这是一个具有表现力的线性注意力模块，扩展了具有更细粒度门控机制的Gated DeltaNet，从而更有效地利用有限的有限状态RNN内存。我们的定制块算法通过一种特殊的对角加低秩（DPLR）变换矩阵实现高硬件效率，与通用DPLR公式相比，显著减少了计算量，同时与经典的delta规则保持更一致。\n\n我们预训练了一个Kimi Linear模型，拥有30亿个激活参数和48亿个总参数，基于KDA和多头潜在注意力（MLA）的逐层混合。我们的实验表明，在相同的训练食谱下，Kimi Linear在所有评估任务中都以显著优势超越了全MLA，同时将KV缓存的使用减少了多达75%，在1M上下文下实现了高达6倍的解码吞吐量。这些结果表明，Kimi Linear可以作为全注意力架构的即插即用替代品，提供更优越的性能和效率，包括处理更长输入和输出长度的任务。\n\n为了支持进一步的研究，我们开源了KDA内核和vLLM实现，并发布了预训练和指令调优的模型检查点。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 54,
          "last_7_days": 54
        },
        "public_total_votes": 10
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
      "abstract": "大型语言模型（LLMs）在需要多步推理的问题上常常面临困难。对于小规模的开源模型，使用可验证奖励的强化学习（RLVR）在经过多次尝试后仍然无法成功采样到正确的解决方案，而监督微调（SFT）则倾向于通过严格的逐词模仿过拟合于长示范。为了填补这一空白，我们提出了监督强化学习（SRL），这是一种将问题解决重新表述为生成逻辑“动作”序列的框架。SRL训练模型在每个动作前生成内部推理独白。它根据模型的动作与从SFT数据集中提取的专家动作之间的相似性，以逐步的方式提供更平滑的奖励。即使所有的回合都是错误的，这种监督也提供了更丰富的学习信号，同时鼓励专家示范引导的灵活推理。结果，SRL使得小模型能够学习以前无法通过SFT或RLVR学习的挑战性问题。此外，在用RLVR进行精细化之前用SRL初始化训练，产生了最强的整体性能。除了推理基准测试之外，SRL在自主软件工程任务中也有效地推广，确立了它作为面向推理的LLM的一个稳健且多功能的训练框架。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 53,
          "last_7_days": 53
        },
        "public_total_votes": 9
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
      "id": "019a29d3-5ad4-745c-89dd-4b5f2a90d263",
      "paper_group_id": "019a29d3-5ad4-745c-89dd-4b5f2a90d263",
      "title": "A Survey of Data Agents: Emerging Paradigm or Overstated Hype?",
      "abstract": "大型语言模型（LLMs）的快速发展促使数据代理的出现——一种旨在协调数据与人工智能生态系统，以应对复杂数据相关任务的自主系统。然而，“数据代理”这一术语目前存在术语模糊和不一致的使用问题，将简单的查询响应者与复杂的自主架构混为一谈。这种术语模糊导致了用户期望的不匹配、责任挑战和行业增长的障碍。受到SAE J3016自动驾驶标准的启发，本调查介绍了首个系统化的阶层分类法，涵盖六个级别，划定并追踪自主性从手动操作（L0）到生成型、完全自主数据代理（L5）的逐步转变，从而澄清能力边界和责任分配。在这一视角下，我们提供了一项结构化的现有研究回顾，按增加的自主性排列，涵盖用于数据管理、准备和分析的专门数据代理，以及朝向具有更高自主性的多功能综合系统的最新努力。我们还分析了推动数据代理发展的关键演变飞跃和技术差距，特别是正在进行的L2到L3的过渡，在这一过程中，数据代理从程序执行演变为自主协调。最后，我们以一个前瞻性的路线图结束，展望积极主动、生成型数据代理的到来。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 210,
          "last_7_days": 210
        },
        "public_total_votes": 24
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
      "id": "019a2df4-d452-707d-b69b-aa09894baaa3",
      "paper_group_id": "019a2df4-d452-707d-b69b-aa09894baaa3",
      "title": "Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs",
      "abstract": "虽然多模态大型语言模型（MLLMs）在视觉理解方面表现出色，但它们常常在需要视觉规划和想象的复杂场景中遇到困难。受到人类通过素描作为一种视觉思维形式来发展和传达想法的启发，我们引入了潜在素描板（Latent Sketchpad），这是一个为MLLMs提供内部视觉草图的框架。MLLMs的内部视觉表示传统上局限于感知理解。我们对其进行了重新利用，以支持生成视觉思维，而不妨碍推理能力。在先进的MLLMs基础上，我们的方法将视觉生成直接整合到其本土自回归推理过程中，使模型能够将文本推理与视觉潜变量的生成交错进行。这些潜变量指导内部思维过程，并且可以转化为草图图像以提高可解释性。为实现这一目标，我们引入了两个组件：一个上下文感知视觉头（Context-Aware Vision Head），自回归地生成视觉表示；一个预训练的素描解码器（Sketch Decoder），将这些表示渲染为人类可理解的图像。我们在新的数据集MazePlanning上评估了这一框架。通过对不同MLLMs的实验表明，潜在素描板的推理性能与其基础模型相当甚至更优。此外，它还可以在不同的前沿MLLMs上推广，包括Gemma3和Qwen2.5-VL。通过将模型的文本推理扩展到视觉思维，我们的框架为更加丰富的人机交互和更广泛的应用开辟了新的机会。更多详细信息和资源可以在我们的项目页面找到：这个网址。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 112,
          "last_7_days": 112
        },
        "public_total_votes": 16
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
      "id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "paper_group_id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "title": "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation",
      "abstract": "大语言模型（LLMs）的近期成功重新引发了人们对推荐系统是否能实现类似规模效益的兴趣。传统推荐系统以庞大的嵌入表为主，这导致当嵌入维度增长时，性能通常趋于平稳。相比之下，新兴的生成范式用自回归Transformer生成的紧凑的语义ID（SID）序列取代了嵌入。然而，大多数工业应用仍然是专有的，这留下了两个基本问题： (1) 预期的规模法则在公共基准上是否成立？ (2) 实现竞争性性能的最小后训练方案是什么？\n\n我们推出了MiniOneRec，尽我们所知，这是第一个完全开源的生成推荐框架，提供了涵盖SID构建、监督微调和面向推荐的强化学习的端到端工作流程。我们通过残差量化变分自编码器生成SID，并在亚马逊评论数据集上对从5亿到70亿参数的Qwen骨干网络进行后训练。我们的实验发现，随着模型规模的增加，训练和评估损失均呈现出一致的下降趋势，验证了生成方法的参数效率。为了进一步提高性能，我们提出了一种轻量且有效的后训练流程，该流程（1）强制执行全流程的SID对齐，(2) 应用有限解码和混合奖励的强化学习。这些技术共同显著提高了排名准确性和候选多样性。",
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
          "all": 137,
          "last_7_days": 137
        },
        "public_total_votes": 21
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
      "id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "paper_group_id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "title": "DeepAgent: A General Reasoning Agent with Scalable Toolsets",
      "abstract": "大型推理模型展示了强大的问题解决能力，但现实世界的任务往往需要外部工具和长时间的互动。现有的代理框架通常遵循预定义的工作流程，这限制了自主和全面任务的完成。在本文中，我们介绍了DeepAgent，一种端到端的深度推理代理，它在单一且一致的推理过程中进行自主思考、工具发现和行动执行。为了解决长时间互动的挑战，特别是多个工具调用导致的上下文长度爆炸和互动历史的累积，我们引入了一种自主记忆折叠机制，将过去的互动压缩成结构化的情节记忆、工作记忆和工具记忆，从而减少错误累积，同时保留关键信息。为了有效且稳定地教授通用工具的使用，我们开发了一种端到端的强化学习策略，即ToolPO，利用LLM模拟的API，并应用工具调用优势归因，将细粒度的信用分配给工具调用的标记。在八个基准测试上的广泛实验，包括通用工具使用任务（ToolBench、API-Bank、TMDB、Spotify、ToolHop）和下游应用（ALFWorld、WebShop、GAIA、HLE），表明DeepAgent在标记工具和开放集工具检索场景中始终优于基线。这项工作朝着为现实世界应用创造更通用和更强大代理的方向迈出了一步。代码和演示可在此HTTPS网址获得。",
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
        "total_votes": 32,
        "visits_count": {
          "all": 1596,
          "last_7_days": 1596
        },
        "public_total_votes": 108
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
      "id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "paper_group_id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "title": "The Principles of Diffusion Models",
      "abstract": "本专著呈现了指导扩散模型发展的核心原则，追溯其起源并展示如何从共同的数学思想中产生多样化的表达形式。扩散建模首先定义一个前向过程，该过程逐渐将数据腐蚀为噪声，通过一系列连续的中间分布将数据分布与简单的先验联系起来。其目标是学习一个反向过程，将噪声转回数据，同时恢复相同的中间过程。我们描述了三种互补的视角。变分视角借鉴变分自编码器，认为扩散是学习逐步去除噪声。基于分数的视角根植于基于能量的建模，学习不断演变的数据分布的梯度，指示如何将样本推向更可能的区域。基于流的视角与标准化流有关，将生成过程视为遵循一条平滑路径，该路径在学习到的速度场下将样本从噪声移动到数据。这些视角共享一个共同的框架：一个时间依赖的速度场，其流动将简单的先验传输到数据。采样因此可以看作是解决一个沿持续轨迹将噪声演变为数据的微分方程。在这个基础上，专著讨论了可控生成的指导方案、高效数值求解器，以及受扩散启发的流映射模型，这些模型学习任意时间之间的直接映射。它为具备基本深度学习知识的读者提供了对扩散模型的概念性和数学基础的理解。",
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
        "total_votes": 39,
        "visits_count": {
          "all": 904,
          "last_7_days": 904
        },
        "public_total_votes": 86
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
      "id": "019a2c33-6b53-74e7-9272-fb16dadac30a",
      "paper_group_id": "019a2c33-6b53-74e7-9272-fb16dadac30a",
      "title": "Towards Personalized Treatment Plan: Geometrical Model-Agnostic Approach to Counterfactual Explanations",
      "abstract": "在我们的文章中，我们描述了一种在高维空间中生成反事实解释的方法，该方法涉及四个步骤：将我们的数据集拟合到模型、找到决策边界、确定问题的约束以及计算该边界上最近的点（反事实解释）。我们提出了一种离散化的方法，在边界上找到许多离散点，然后识别最近的可行反事实解释。我们稍后称之为“分段采样边界近似”（SSBA）的方法，应用二分搜索找到决策边界点，然后寻找最近的边界点。在四个不同维度的数据集上，我们展示了我们的方法可以超越当前的反事实生成方法，在$L_2$范数方面的距离减少幅度在$5\\%$到$50\\%$之间。我们的方法还可以通过对不可变和分类特征（如年龄、性别、性别、高度以及与健康相关的数据集相似的其他相关特征）限制变化，从而处理现实世界的约束。在运行时间方面，SSBA算法在给定时间内生成决策边界点的效率比基于网格的方法快多个数量级。总体而言，我们的方法提供了一种简单有效的模型无关方法，能够计算出最近的可行（即符合约束的现实）反事实解释。我们的所有结果和代码可以在此链接找到：$\\href{this https URL}{this https URL dsin85691/SSBA\\_For\\_Counterfactuals}$",
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
          "all": 99,
          "last_7_days": 99
        },
        "public_total_votes": 13
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
      "id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "paper_group_id": "019a33b3-1cb4-740e-9f6f-5301838f9711",
      "title": "Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks",
      "abstract": "人类拥有空间推理能力，使他们能够通过视觉和声音等多种感官观察理解空间。大型多模态推理模型通过学习感知和推理扩展了这些能力，在各种空间任务中展示出良好的性能。然而，针对这些模型的系统评审和公开基准仍然有限。在这项调查中，我们对大模型的多模态空间推理任务进行了全面评审，归纳了多模态大型语言模型（MLLM）的最新进展，并介绍了用于评估的开放基准。我们首先概述了一般的空间推理，重点讨论后训练技术、可解释性和架构。除了经典的二维任务外，我们还考察了空间关系推理、场景和布局理解，以及视觉问答和在三维空间中的基础定位。我们还审视了体现人工智能的进展，包括视觉-语言导航和动作模型。此外，我们考虑了音频和第一人称视频等新兴模态，它们通过新传感器为新颖的空间理解做出了贡献。我们相信这项调查为不断增长的多模态空间推理领域奠定了坚实的基础，并提供了有价值的见解。有关此调查的最新信息、代码和开放基准的实现可在此https网址找到。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 54,
          "last_7_days": 54
        },
        "public_total_votes": 11
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
      "id": "019a3258-ba90-763b-b54f-f9819b50e6d9",
      "paper_group_id": "019a3258-ba90-763b-b54f-f9819b50e6d9",
      "title": "Generating Creative Chess Puzzles",
      "abstract": "尽管生成式人工智能在各个领域迅速发展，但生成真正具有创造性、美感和反直觉的输出仍然是一项挑战。本文提出了一种应对国际象棋难题领域中这些困难的方法。我们首先对生成式人工智能架构进行基准测试，然后引入一个基于国际象棋引擎搜索统计的创新奖励的强化学习框架，以克服一些缺陷。这些奖励旨在提升难题的独特性、反直觉性、多样性和现实感。我们的强化学习方法将反直觉难题的生成提升了10倍，从0.22\\%（监督学习）提高到2.5\\%，超过了现有数据集的比率（2.1\\%）和最佳的Lichess训练模型（0.4\\%）。我们的难题达到了新颖性和多样性基准，保留了美学主题，并获得人类专家的评价，认为其比书中编撰的难题更具创造性、趣味性和反直觉性，甚至接近经典作品。我们的最终成果是一本经过精选的AI生成难题小册子，该小册子获得了三位世界著名专家对创造力的认可。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 68,
          "last_7_days": 68
        },
        "public_total_votes": 13
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
      "id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "paper_group_id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "title": "Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents",
      "abstract": "我们推出了Game-TARS，作为一个通用游戏代理，通过统一、可扩展的动作空间进行训练，该动作空间以与人类对齐的原生键盘鼠标输入为基础。与基于API或GUI的方法不同，这种范式支持在异构领域进行大规模持续的预训练，包括操作系统、网络和模拟游戏。Game-TARS在超过5000亿个标记上进行预训练，涵盖了多样的轨迹和多模态数据。关键技术包括一种衰减持续损失，以减少因果混淆，以及一种高效的稀疏思维策略，以平衡推理深度和推断成本。实验表明，Game-TARS在开放世界的Minecraft任务中成功率约为之前最先进模型的两倍，在未见过的网络3D游戏中接近新手玩家的普遍性，并且在FPS基准测试中超越了GPT-5、Gemini-2.5-Pro和Claude-4-Sonnet。关于训练时间和测试时间的扩展结果确认，统一的动作空间在跨游戏和多模态数据扩展时持续带来改进。我们的结果表明，简单、可扩展的动作表示结合大规模预训练，为拥有广泛计算机使用能力的通用代理提供了一条有希望的道路。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 113,
          "last_7_days": 113
        },
        "public_total_votes": 16
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
      "id": "019a3409-76c2-72e8-a9d2-1c507e3a57b5",
      "paper_group_id": "019a3409-76c2-72e8-a9d2-1c507e3a57b5",
      "title": "JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence",
      "abstract": "神经编码智能的范围正在迅速扩展，超越基于文本的源代码，涵盖程序生成的丰富视觉输出。这一视觉维度对于灵活的内容生成以及精确的程序驱动可视化编辑等高级应用至关重要。然而，进展受到高质量多模态代码数据稀缺的阻碍，这一瓶颈源于合成和质量评估的挑战。为了解决这些挑战，我们从数据和建模两个角度做出了贡献。我们首先介绍了一整套合成工具包，利用数据模态之间的相互协同，高效地生成从标准图表到复杂交互式网页用户界面及代码驱动动画的大规模高质量语料库。基于此工具包，我们构建了JanusCode-800K，这是迄今为止最大的多模态代码语料库。这为我们的模型JanusCoder和JanusCoderV的训练提供了支持，后者建立了一个视觉编程接口，用于从文本指令、视觉输入或两者的组合生成代码。我们的统一模型不同于现有的为孤立任务构建专用模型的方法。在文本中心和视觉中心的编码任务上进行的广泛实验表明，JanusCoder系列的性能优越，我们的7B至14B规模模型的表现接近甚至超过商业模型。此外，广泛的分析提供了统一程序逻辑与其视觉表达的关键见解。我们的代码和检查点将在此https URL上提供。",
      "paper_summary": {
        "summary": "JANUSCODER presents a unified visual-programmatic interface for multimodal code intelligence, addressing the gap between code's logical structure and its visual output. It introduces JANUSCODE-800K, the largest multimodal code corpus, and achieves performance matching or surpassing commercial models like GPT-4o on diverse text-centric and vision-centric coding tasks.",
        "originalProblem": [
          "Existing research in visually-grounded code generation and understanding is fragmented, relying on specialized models for isolated tasks that lack generalization.",
          "Progress in multimodal code intelligence is bottlenecked by the scarcity of high-quality, diverse multimodal code data, particularly for complex visual outputs like animations and interactive UIs."
        ],
        "solution": [
          "A novel, scalable data synthesis toolkit was developed to automatically generate high-quality multimodal code data through strategies like Guided Evolution, Re-Contextualization, and Bidirectional Translation.",
          "The JANUSCODE-800K corpus, comprising 800,000 diverse samples including animations and web artifacts, was constructed as the largest and most comprehensive multimodal code dataset.",
          "JANUSCODER (text-centric) and JANUSCODERV (multimodal) models were trained on this corpus to unify the generation of code from textual and visual inputs, or combinations thereof."
        ],
        "keyInsights": [
          "Cross-domain and cross-modal data synergies are critical for enhancing model performance, enabling knowledge transfer, and improving capabilities in data-scarce scenarios.",
          "A VLM/LLM-based reward modeling framework is crucial for filtering out misaligned or low-quality synthesized data, as mere executability is insufficient for high-quality multimodal data.",
          "The proposed data design and synthesis toolkit effectively empower various large language model (LLM) and vision-language model (VLM) backbones, leading to consistent performance improvements across tasks."
        ],
        "results": [
          "JANUSCODE-800K was successfully created as the largest multimodal code corpus, offering a balanced distribution of text-centric (50.9%) and vision-centric (49.1%) data.",
          "JANUSCODER and JANUSCODERV models demonstrated superior performance, often matching or exceeding commercial models like GPT-4o, on Python visualization, web artifact generation, and scientific demonstration tasks.",
          "On multimodal chart-to-code tasks, JANUSCODERV significantly outperformed baselines and specialized MLLMs, validating the effectiveness of cross-task data synergy."
        ]
      },
      "image_url": "image/2510.23538v1.png",
      "universal_paper_id": "2510.23538",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 54,
          "last_7_days": 54
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-27T17:13:49.000Z",
      "publication_date": "2025-10-27T17:13:49.000Z",
      "updated_at": "2025-10-30T07:33:46.818Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.SE",
        "generative-models",
        "multi-modal-learning",
        "representation-learning",
        "synthetic-data",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 31,
      "github_url": "https://github.com/InternLM/JanusCoder",
      "distance": 1
    },
    {
      "id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "paper_group_id": "019a331e-0ea8-7871-9ff9-ada99cb7db45",
      "title": "Parallel Loop Transformer for Efficient Test-Time Computation Scaling",
      "abstract": "大型语言模型（LLMs）功能强大，但在推理过程中往往太慢且成本高，不适合实际应用。循环变压器通过在多个计算步骤或“循环”中重用相同的权重来节省参数。然而，这种方法有一个主要缺陷：循环是依次运行的，每增加一个循环，推理延迟和内存需求就会增加。这使得它们在快速应用中变得不切实际。为了解决这个问题，我们提出了并行循环变压器（PLT）。PLT是一种新架构，能够提供深度循环模型的性能优势，同时具备标准非循环模型的低延迟。PLT通过两种关键技术实现。首先，交叉循环并行性（CLP）通过同时计算不同标记的不同循环打破了顺序依赖，全部在一次传递中完成。其次，为了防止内存成本增加，我们采用高效表示增强策略。该方法将第一个循环的内存（KV缓存）与所有其他循环共享。然后，它使用门控滑动窗口注意力（G-SWA）将这种共享的全局信息与局部信息结合，保持高准确性。我们的实验表明，PLT在与标准变压器比较时，几乎没有额外的延迟或内存成本，同时能够达到传统循环模型的高准确性。",
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
          "all": 63,
          "last_7_days": 63
        },
        "public_total_votes": 13
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
      "id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "paper_group_id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "title": "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations",
      "abstract": "人类通过多感官的协同作用学习抽象概念，一旦形成，这种表征通常可以通过单一模态进行回忆。受到这一原理的启发，我们推出了Concerto，这是一个极简的空间认知人类概念学习模拟，结合了3D内模态自蒸馏和2D-3D跨模态联合嵌入。尽管其简单，Concerto仍能学习到更连贯和信息量更大的空间特征，零-shot可视化证明了这一点。在3D场景感知的线性探测中，它分别比独立的SOTA 2D和3D自监督模型提高了14.2%和4.8%，以及它们的特征拼接。通过全面微调，Concerto在多个场景理解基准上创下了新的SOTA结果（例如，ScanNet上的80.7% mIoU）。我们进一步展示了一个针对视频提升点云空间理解的Concerto变体，以及一个将Concerto表示线性投影到CLIP语言空间的转换器，实现开放世界感知。这些结果突出了Concerto在空间表征中展现出优越的细粒度几何和语义一致性。",
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
        "total_votes": 4,
        "visits_count": {
          "all": 212,
          "last_7_days": 212
        },
        "public_total_votes": 29
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
      "id": "019a330e-e61d-7672-97ca-c751585d6abc",
      "paper_group_id": "019a330e-e61d-7672-97ca-c751585d6abc",
      "title": "Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation",
      "abstract": "我们提出了Ming-Flash-Omni，这是Ming-Omni的升级版，基于稀疏的专家混合（MoE）变体Ling-Flash-2.0构建，拥有1000亿总参数，每个token仅激活61亿。这种架构实现了高效的扩展（显著提高了计算效率，同时大幅扩大了模型容量），并赋予了跨视觉、语音和语言的更强大的统一多模态智能，代表了向人工通用智能（AGI）迈出的重要一步。与其前身相比，升级版在多模态理解和生成方面表现出显著改善。我们在语音识别能力上有了重大进展，在上下文自适应语音识别（ASR）中取得了行业领先的表现，并在方言感知ASR中表现出高度竞争力。在图像生成方面，Ming-Flash-Omni引入了高保真文本渲染，并在图像编辑过程中展示了场景一致性和身份保持的显著提升。此外，Ming-Flash-Omni引入了生成分割，这一能力不仅实现了强大的独立分割性能，还增强了图像生成中的空间控制，改善了编辑一致性。值得注意的是，Ming-Flash-Omni在文本到图像生成和生成分割方面达到了行业领先的结果，并在所有12个上下文ASR基准测试中创下新纪录，所有这些都在一个统一的架构中实现。",
      "paper_summary": {
        "summary": "Ming-Flash-Omni, from Inclusion AI, Ant Group, presents a sparse Mixture-of-Experts architecture with 100 billion total parameters and only 6.1 billion active per token, unifying multimodal perception and generation across vision, speech, and language. The model achieves state-of-the-art performance in contextual ASR and generative segmentation while demonstrating quality comparable to specialized models in text-to-image generation and significant improvements over its predecessor across various multimodal understanding tasks.",
        "originalProblem": [
          "Effectively integrating comprehension and generation across multiple modalities into a unified model remains challenging due to representational disparities and modality imbalances.",
          "Scaling large language models for multimodal tasks typically leads to prohibitive computational and memory costs, hindering real-world deployment and practical applications.",
          "Existing multimodal AI systems often struggle with specialized real-world scenarios, such as highly accurate contextual ASR, dialect-aware recognition, and fine-grained spatial control in image generation."
        ],
        "solution": [
          "The system utilizes a sparse Mixture-of-Experts (MoE) architecture (Ling-Flash-2.0) with 100 billion total parameters, activating only 6.1 billion per token for computational efficiency.",
          "A unified two-stage pipeline integrates a perception stage for multimodal understanding (with VideoRoPE and context-aware ASR) and a generation stage for speech and image synthesis (using continuous acoustic representations and generative segmentation).",
          "Novel training paradigms, including 'generative segmentation as an editing task,' advanced identity preservation, and high-fidelity text rendering, are introduced to bridge understanding and generation and enhance control."
        ],
        "keyInsights": [
          "Sparse MoE architectures allow for a dramatic expansion of model capacity (100 billion parameters) while maintaining high computational efficiency (6.1 billion active parameters per token), making powerful multimodal models practical.",
          "Reformulating segmentation as a 'generative editing task' unifies perception and generation objectives by requiring the model to understand object boundaries as a prerequisite for semantic editing, leading to improved spatio-semantic control.",
          "Employing continuous acoustic representations for speech generation, rather than discrete tokens, significantly improves the naturalness, expressiveness, and overall fidelity of synthesized speech."
        ],
        "results": [
          "Achieved state-of-the-art performance across all 12 contextual ASR benchmarks and set new SOTA records in generative segmentation, significantly outperforming other unified MLLMs.",
          "Delivered state-of-the-art results in text-to-image generation on GenEval with exceptional controllability (particularly in 'Position' and 'Color') and quality comparable to specialized image generation models like SD3-Medium.",
          "Demonstrated highly competitive or leading performance across a broad spectrum of multimodal understanding benchmarks, including SOTA on MVBench for video reasoning and strong results in audio QA, visual-text understanding, and OCR."
        ]
      },
      "image_url": "image/2510.24821v1.png",
      "universal_paper_id": "2510.24821",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 53,
          "last_7_days": 53
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-28T15:24:13.000Z",
      "publication_date": "2025-10-28T15:24:13.000Z",
      "updated_at": "2025-10-30T03:00:05.789Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "efficient-transformers",
        "generative-models",
        "image-generation",
        "image-segmentation",
        "multi-modal-learning",
        "speech-recognition",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "Ant Group",
          "image": null
        },
        {
          "name": "Inclusion AI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 499,
      "github_url": "https://github.com/inclusionAI/Ming",
      "distance": 1
    },
    {
      "id": "019a2e12-fcf3-7eff-849d-a4ee1116b394",
      "paper_group_id": "019a2e12-fcf3-7eff-849d-a4ee1116b394",
      "title": "Video-Thinker: Sparking \"Thinking with Videos\" via Reinforcement Learning",
      "abstract": "最近在图像推理方法方面的进展，特别是“以图像思考”，在多模态大型语言模型（MLLM）中取得了显著成功；然而，这种动态推理范式尚未扩展到视频推理任务。在本文中，我们提出了Video-Thinker，使MLLM能够通过自主利用其固有的“基础”与“字幕”能力，在推理过程中生成推理线索。为了激发这一能力，我们构建了Video-Thinker-10K，这是一个策划的数据集，展示了在思维链推理序列中自主工具使用。我们的训练策略首先是监督微调（SFT），以学习推理格式，然后通过群体相对策略优化（GRPO）来增强这一推理能力。通过这种方法，Video-Thinker使得MLLM能够自主处理视频推理中的基础和字幕任务，无需构建和调用外部工具。大量实验表明，Video-Thinker在领域内任务和具有挑战性的领域外视频推理基准（包括Video-Holmes、CG-Bench-Reasoning和VRBench）上都获得了显著的性能提升。我们的Video-Thinker-7B大幅超过现有基线如Video-R1，并在7B规模的MLLM中树立了最先进的性能。",
      "paper_summary": {
        "summary": "VIDEO-THINKER, a new framework, empowers Multimodal Large Language Models to reason with videos by intrinsically developing temporal grounding and captioning abilities. The model establishes new state-of-the-art performance on various video reasoning benchmarks, achieving up to an 11.44% improvement on the VRBench out-of-domain dataset, while showcasing enhanced temporal localization (48.22% mIoU) and descriptive captioning.",
        "originalProblem": [
          "Extending the 'Thinking with Images' paradigm to videos poses challenges due to inherent temporal dependencies and dynamic content.",
          "Existing MLLMs often treat videos as static inputs or rely on rigid, pre-designed prompting strategies, limiting dynamic manipulation and reasoning over temporal sequences.",
          "Many video reasoning models struggle to seamlessly integrate temporal grounding into the full chain-of-thought process and typically demand extensive datasets."
        ],
        "solution": [
          "A Hindsight-Curation Reasoning pipeline synthesizes the Video-Thinker-10K dataset, providing structured reasoning traces with explicit temporal (`<time>`), captioning (`<caption>`), and thinking (`<think>`) tags.",
          "A two-stage training strategy is employed, beginning with Supervised Fine-Tuning (SFT) to teach the MLLM to generate reasoning in the specified format.",
          "The SFT stage is followed by Group Relative Policy Optimization (GRPO), a reinforcement learning technique, to intrinsically strengthen grounding and captioning capabilities for autonomous temporal navigation and robust reasoning."
        ],
        "keyInsights": [
          "Empowering MLLMs with intrinsic temporal grounding and captioning allows them to autonomously 'think with videos,' moving beyond passive video processing to active, dynamic reasoning.",
          "The hindsight-curation data synthesis, combined with the two-stage SFT and GRPO training, efficiently instills complex video reasoning abilities using a relatively small 10K sample dataset.",
          "The GRPO stage is crucial for developing robust reasoning and out-of-domain generalization, transforming basic format-following into effective autonomous temporal navigation and understanding."
        ],
        "results": [
          "Video-Thinker-7B achieves new state-of-the-art results among 7B-sized MLLMs across in-domain and challenging out-of-domain video reasoning benchmarks, with up to an 11.44% improvement on VRBench.",
          "The model demonstrates superior intrinsic grounding abilities with an mIoU of 48.22% (a 75.5% relative improvement over Qwen2.5-VL-7B) and enhanced captioning, showing a 31.2% relative enhancement in overall captioning performance.",
          "Video-Thinker exhibits emergent 'aha moments' and self-reflective behaviors during reasoning, suggesting dynamic internal feedback mechanisms and metacognitive capabilities acquired with high data efficiency."
        ]
      },
      "image_url": "image/2510.23473v1.png",
      "universal_paper_id": "2510.23473",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 65,
          "last_7_days": 65
        },
        "public_total_votes": 15
      },
      "first_publication_date": "2025-10-27T16:10:45.000Z",
      "publication_date": "2025-10-27T16:10:45.000Z",
      "updated_at": "2025-10-29T03:46:27.699Z",
      "topics": [
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.CV",
        "deep-reinforcement-learning",
        "fine-tuning",
        "multi-modal-learning",
        "tool-use",
        "transformers",
        "video-understanding",
        "vision-language-models",
        "visual-reasoning"
      ],
      "organization_info": [
        {
          "name": "Monash University",
          "image": "images/organizations/monash-university.png"
        },
        {
          "name": "University of Southern California",
          "image": "images/organizations/usc.png"
        },
        {
          "name": "Fudan University",
          "image": "images/organizations/fudan-university.png"
        },
        {
          "name": "Southeast University",
          "image": null
        },
        {
          "name": "Xiaohongshu Inc.",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 20,
      "github_url": "https://github.com/shijian2001/Video-Thinker",
      "distance": 1
    },
    {
      "id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "paper_group_id": "019a34ca-6e21-70d7-9aae-8fe55a4b67a2",
      "title": "GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning",
      "abstract": "由大型语言模型（LLMs）驱动的自主智能体在复杂任务解决中的工具操控表现出了令人印象深刻的能力。然而，现有的范式如ReAct依赖顺序推理和执行，未能充分利用独立子任务之间固有的并行性。这一顺序瓶颈导致工具利用效率不高，在多步推理场景下表现不佳。我们提出了基于图的智能体规划（GAP），这一新型框架通过图形规划明确建模任务之间的依赖关系，以实现自适应的并行和串行工具执行。我们的方法训练智能体基础模型，将复杂任务分解为关注依赖关系的子任务图，自主确定哪些工具可以并行执行，哪些工具必须遵循顺序依赖。这种关注依赖关系的协调在执行效率和任务准确性上都取得了显著提升。为了训练GAP，我们构建了一个高质量的数据集，该数据集由多跳问答（MHQA）基准中的图基规划痕迹派生而来。我们采用两阶段的训练策略：在策划好的数据集上进行监督微调（SFT），然后在有战略性抽样查询的基础上，使用以正确性为基础的奖励函数进行强化学习（RL），其中基于工具的推理提供最大价值。在MHQA数据集上的实验结果表明，GAP显著优于传统的ReAct基线，特别是在多步检索任务中，同时通过智能并行化实现了工具调用效率的显著提高。项目页面可在此链接找到：这个 https URL。",
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
          "all": 27,
          "last_7_days": 27
        },
        "public_total_votes": 4
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
      "id": "019a298d-625b-7589-9dae-bd0721cafa20",
      "paper_group_id": "019a298d-625b-7589-9dae-bd0721cafa20",
      "title": "ReCode: Unify Plan and Action for Universal Granularity Control",
      "abstract": "现实世界的任务要求在不同的粒度上作出决策，人类通过利用统一的认知表征在这一点上表现出色，其中规划基本上被理解为一种高层次的行动。然而，当前基于大型语言模型（LLM）的代理缺乏这种在决策粒度之间流畅操作的关键能力。这一限制源于现有范式强制对高层规划和低层行动进行严格分隔，进而削弱了动态适应性并限制了推广能力。我们提出了ReCode（递归代码生成），一种新颖的范式，通过将规划和行动统一在一个代码表征中，解决了这一限制。在这种表征中，ReCode将高层计划视为抽象占位函数，代理然后递归地将其分解为更细粒度的子函数，直到达到原始动作。这种递归方法消除了计划与行动之间的严格界限，使代理能够动态控制其决策粒度。此外，递归结构本质上生成丰富的多粒度训练数据，使模型能够学习层级决策过程。大量实验表明，ReCode在推理性能上显著超越了先进基线，并在训练中展现出卓越的数据效率，验证了我们核心见解：通过递归代码生成将规划和行动统一起来是一种强大而有效的方法，可以实现普遍的粒度控制。代码可在此 https URL 获取。",
      "paper_summary": {
        "summary": "DeepWisdom, alongside researchers from The Hong Kong University of Science and Technology (Guangzhou), Renmin University of China, Zhejiang University, and Mila, developed RECODE, a paradigm that unifies planning and action in Large Language Model agents through recursive code generation. This approach provides agents with universal granularity control, leading to a 20.9% relative improvement in average task reward over state-of-the-art baselines and a substantial reduction in both training data requirements and inference costs.",
        "originalProblem": [
          "Existing LLM-based agents, like ReAct, operate with a fixed, low level of decision granularity, limiting strategic foresight and adaptability in complex tasks.",
          "Agents with explicit planners rigidly separate high-level planning from low-level action execution, preventing dynamic adjustment of decision granularity based on evolving task complexities.",
          "These architectural constraints lead to brittle performance and reduced generalization capabilities in dynamic, real-world environments."
        ],
        "solution": [
          "RECODE unifies planning and action by representing both as Python function calls, treating abstract plans as higher-level actions or 'placeholder functions'.",
          "It employs a recursive generation-execution loop where the LLM policy dynamically decomposes high-level placeholder functions into finer-grained sub-functions until primitive, executable actions are reached.",
          "This process allows agents to build a hierarchical decision tree, adapting decision granularity to the current context without explicit supervision."
        ],
        "keyInsights": [
          "Planning and action are not distinct cognitive processes but rather represent decisions at different levels of granularity within a unified control framework.",
          "Representing plans as abstract placeholder functions that are recursively expanded enables dynamic and universal control over decision granularity, mirroring human cognitive flexibility.",
          "The inherent hierarchical structure of recursive code generation naturally yields rich, multi-granularity training data, which significantly enhances learning signals and data efficiency for agents."
        ],
        "results": [
          "RECODE achieved an average reward of 60.8% across ALFWorld, WebShop, and ScienceWorld, outperforming the best baseline (AdaPlanner) by 10.5% (a 20.9% relative improvement).",
          "It demonstrated superior data efficiency, achieving strong performance (70.4% with Qwen2.5-7B-Instruct) with significantly less training data than ReAct+SFT and CodeAct+SFT.",
          "The paradigm showed remarkable cost efficiency, with RECODE trajectories costing 78.9% less than ReAct and 84.4% less than CodeAct on average, due to more structured exploration and potent decisions."
        ]
      },
      "image_url": "image/2510.23564v1.png",
      "universal_paper_id": "2510.23564",
      "metrics": {
        "total_votes": 8,
        "visits_count": {
          "all": 118,
          "last_7_days": 118
        },
        "public_total_votes": 25
      },
      "first_publication_date": "2025-10-27T17:35:15.000Z",
      "publication_date": "2025-10-27T17:35:15.000Z",
      "updated_at": "2025-10-28T06:42:02.971Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "data-curation",
        "reasoning",
        "representation-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 4,
      "github_url": "https://github.com/FoundationAgents/ReCode",
      "distance": 1
    }
  ],
  "page": 0
};