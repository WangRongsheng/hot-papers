const papersData = {
  "papers": [
    {
      "id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "paper_group_id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "title": "DeepAgent: A General Reasoning Agent with Scalable Toolsets",
      "abstract": "大型推理模型展示了强大的问题解决能力，但现实世界的任务往往需要外部工具和长时间的交互。现有的智能体框架通常遵循预定义的工作流程，这限制了自主和全局任务的完成。在本文中，我们介绍了DeepAgent，这是一种端到端的深度推理智能体，在单一、连贯的推理过程中进行自主思考、工具发现和行动执行。为了解决长时间交互带来的挑战，特别是多次工具调用引起的上下文长度爆炸和交互历史的累积，我们引入了一种自主记忆折叠机制，将过去的交互压缩为结构化的情节记忆、工作记忆和工具记忆，在保留关键信息的同时减少错误累积。为了高效而稳定地教授通用工具使用，我们开发了一种端到端的强化学习策略，即ToolPO，该策略利用LLM模拟的API并应用工具调用优势归因，以便为工具调用令牌分配细致的信用。在八个基准测试上的大量实验，包括通用工具使用任务（ToolBench、API-Bank、TMDB、Spotify、ToolHop）和下游应用（ALFWorld、WebShop、GAIA、HLE），显示DeepAgent在标签工具和开放集合工具检索场景中始终优于基线。该工作向实现更通用和更强大的智能体用于现实世界应用迈出了重要一步。代码和演示可以在此HTTPS URL获取。",
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
        "total_votes": 23,
        "visits_count": {
          "all": 969,
          "last_7_days": 969
        },
        "public_total_votes": 69
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
      "id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "paper_group_id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "title": "Real Deep Research for AI, Robotics and Beyond",
      "abstract": "随着人工智能和机器人研究的快速增长，每年产生超过10,000篇论文，研究人员保持更新的难度越来越大。迅速发展的趋势、跨学科工作的崛起以及探索超出自己专业领域的需求，都对这一挑战贡献良多。为了解决这些问题，我们提出了一种可泛化的流程，能够系统地分析任何研究领域：识别新兴趋势、发掘跨领域机会，并为新的研究提供具体的起点。在这项工作中，我们呈现了“真实深度研究”（Real Deep Research，简称RDR），这是一个应用于人工智能和机器人领域的综合框架，特别关注基础模型和机器人技术的进展。我们还简要扩展了对其他科学领域的分析。主要论文详细介绍了RDR流程的构建，而附录则提供了每个分析主题的广泛结果。我们希望这项工作能为在人工智能及其相关领域工作的研究人员提供启示。",
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
        "total_votes": 28,
        "visits_count": {
          "all": 1399,
          "last_7_days": 1399
        },
        "public_total_votes": 103
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
      "id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "paper_group_id": "019a2c20-a499-781f-b895-e0c2dc989b15",
      "title": "Multi-Agent Evolve: LLM Self-Improve through Co-evolution",
      "abstract": "强化学习（RL）在提升大型语言模型（LLMs）的推理能力方面展示了显著潜力。然而，RL在LLMs中的成功在很大程度上依赖于人工策划的数据集和可验证的奖励，这限制了其可扩展性和普适性。最近受游戏和围棋成功启发的自我对弈RL方法，旨在无需人工标注数据来增强LLM的推理能力。然而，它们的方法主要依赖于具有反馈的基础环境（例如，Python解释器或游戏引擎）；将它们扩展到一般领域仍然具有挑战性。为了解决这些挑战，我们提出了多智能体演化（MAE）框架，该框架使LLM能够在解决各种任务中自我演化，包括数学、推理和一般知识问答。MAE的核心设计基于三个相互作用的智能体（提议者、求解者、评判者），它们由单个LLM实例化，并应用强化学习来优化其行为。提议者生成问题，求解者尝试解决方案，评判者在共同演化的过程中评估两者。对Qwen2.5-3B-Instruct的实验表明，MAE在多个基准测试中实现了平均4.54%的提升。这些结果突显出MAE是一种可扩展、数据高效的方法，能够在最小程度上依赖人工策划的监督，从而增强LLM的一般推理能力。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 110,
          "last_7_days": 110
        },
        "public_total_votes": 12
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
      "id": "019a2698-a848-7d12-b559-474738d7624c",
      "paper_group_id": "019a2698-a848-7d12-b559-474738d7624c",
      "title": "Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine",
      "abstract": "最近的研究通过编码代理自我编辑代码库来实现自我改进。它们通过扩展策略生成一棵自我修改的树，这些策略有利于更高的软件工程基准性能，假设这意味着更有前景的后续自我修改。然而，我们发现代理的自我改进潜力（元生产力）与其编码基准性能之间存在不匹配，即元生产力-性能不匹配。受到赫胥黎分类概念的启发，我们提出了一种度量标准（$\\mathrm{CMP}$），汇总代理后代的基准表现，以作为其自我改进潜力的指示。我们表明，在我们的自我改进编码代理开发环境中，获取真实的$\\mathrm{CMP}$足以模拟哥德尔机器在某些假设下的行为。我们引入赫胥黎-哥德尔机器（HGM），该机器通过估计$\\mathrm{CMP}$并将其作为指导，搜索自我修改的树。在SWE-bench Verified和Polyglot上，HGM在使用更少的实际时间的情况下，优于先前的自我改进编码代理开发方法。最后但同样重要的是，HGM在其他编码数据集和大型语言模型上表现出强大的迁移能力。在SWE-bench Verified上，HGM在GPT-5-mini的优化下，并在SWE-bench Lite上用GPT-5进行评估，达到了人类级别的表现，匹配了人类设计编码代理的最佳官方检查结果。我们的代码可在此链接获取。",
      "paper_summary": {
        "summary": "The Huxley-Gödel Machine (HGM) framework introduces Clade-Metaproductivity (CMP) to guide self-improving coding agents, overcoming the Metaproductivity–Performance Mismatch observed in prior methods. This approach enables the discovery of coding agents that achieve human-level performance on SWE-Bench benchmarks with greater efficiency and robustness.",
        "originalProblem": [
          "Existing self-improving AI agents (e.g., DGM, SICA) suffer from a \"Metaproductivity–Performance Mismatch,\" where immediate benchmark performance is a poor indicator of long-term self-improvement potential.",
          "Prior methods rely on greedy heuristics that favor agents with high short-term performance, often leading to evolutionary dead ends rather than lineages with sustained improvement capacity.",
          "Implementing provably optimal self-improvement, as theorized by the Gödel Machine, has been largely impractical due to its reliance on formal proofs."
        ],
        "solution": [
          "HGM formalizes self-improvement as a tree-search problem and introduces Clade-Metaproductivity (CMP), a lineage-based metric that aggregates the success of all an agent's descendants to measure long-term potential.",
          "The algorithm employs a decoupled, three-sub-policy structure: a CMP estimator, a Thompson Sampling-based expansion policy with an adaptive exploration-exploitation scheduler, and a decoupled evaluation policy for efficient resource allocation.",
          "An asynchronous execution model (HGM Async) is implemented to concurrently run multiple iterations, significantly boosting efficiency and reducing wall-clock time."
        ],
        "keyInsights": [
          "Clade-Metaproductivity (CMP) serves as a theoretically robust proxy for long-term self-improvement potential, directly addressing and mitigating the Metaproductivity–Performance Mismatch.",
          "Under specific assumptions, access to a true CMP oracle is sufficient to implement a Gödel Machine in the context of self-improving coding agent development.",
          "Decoupling expansion and evaluation, combined with adaptive search strategies like Thompson Sampling, leads to more efficient and effective exploration of the agent design space."
        ],
        "results": [
          "HGM's estimated CMP showed significantly stronger correlations (0.778 weighted on SWE-Verified-60) with empirical CMP compared to prior methods (0.274-0.444).",
          "On SWE-Verified-60, HGM achieved 56.7% accuracy (+16.7% gain) and required 517 CPU-hours, outperforming DGM (53.3% accuracy, 1231 CPU-hours) and SICA (50.0% accuracy, 572 CPU-hours).",
          "HGM discovered an agent that achieved 61.4% task resolution on the full SWE-Bench Verified dataset, surpassing top human-designed agents using the same LLM backbone and demonstrating strong generalization to SWE-Bench Lite (49.0% on standard, 57.0% with GPT-5)."
        ]
      },
      "image_url": "image/2510.21614v1.png",
      "universal_paper_id": "2510.21614",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 392,
          "last_7_days": 392
        },
        "public_total_votes": 33
      },
      "first_publication_date": "2025-10-24T16:19:41.000Z",
      "publication_date": "2025-10-24T16:19:41.000Z",
      "updated_at": "2025-10-27T16:55:30.120Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "meta-learning",
        "optimization-methods",
        "reasoning",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 8,
      "github_url": "https://github.com/metauto-ai/HGM",
      "distance": 1
    },
    {
      "id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "paper_group_id": "019a2d98-dd10-79a3-8241-720f043f888b",
      "title": "The Principles of Diffusion Models",
      "abstract": "本专著呈现了指导扩散模型发展的核心原则，追溯其起源并展示了多样化的公式是如何源于共享的数学思想。扩散建模首先定义了一个前向过程，该过程逐步将数据腐蚀为噪声，通过一系列中间分布将数据分布与一个简单的先验联系起来。其目标是学习一个反向过程，将噪声转变回数据，同时恢复相同的中间值。我们描述了三种互补的视角。变分视角受到变分自编码器的启发，将扩散视为逐步去除噪声的过程。基于评分的视角根植于基于能量的建模，学习不断演变的数据分布的梯度，指示如何将样本推向更可能的区域。基于流的视角与归一化流相关，将生成视为沿着一个光滑路径移动样本，从噪声到数据，遵循一个学习到的速度场。这些观点共享一个共同的基础：一个时间依赖的速度场，其流动将一个简单的先验传输到数据。采样则相当于解决一个沿着连续轨迹将噪声演变为数据的微分方程。在此基础上，本专著讨论了可控生成的指导、有效的数值求解器以及扩散激励的流映射模型，这些模型学习任意时间之间的直接映射。它为具有基本深度学习知识的读者提供了对扩散模型的概念性和数学基础的理解。",
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
        "total_votes": 12,
        "visits_count": {
          "all": 263,
          "last_7_days": 263
        },
        "public_total_votes": 27
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
      "id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "paper_group_id": "019a28ef-c275-7d0f-b273-32eaa4d8e277",
      "title": "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations",
      "abstract": "人类通过多感官协同学习抽象概念，一旦形成，这种表征通常可以通过单一模态进行回忆。受到这一原则的启发，我们推出了Concerto，一种人类概念学习的极简模拟，旨在提升空间认知，结合了3D内模态自蒸馏和2D-3D跨模态联合嵌入。尽管其简单，Concerto仍能学习更具一致性和信息量的空间特征，这通过零样本可视化得到了证明。在3D场景感知的线性探测中，Concerto分别比独立的最先进的2D和3D自监督模型提高了14.2%和4.8%的性能，并且优于它们的特征连接。通过全面微调，Concerto在多个场景理解基准中创下新的最先进结果（例如，在ScanNet上达到80.7%的mIoU）。我们进一步推出了一个Concerto的变体，专门用于视频提升的点云空间理解，以及一个将Concerto表征线性投影到CLIP语言空间的翻译器，从而实现开放世界感知。这些结果突显了Concerto在空间表征中展现出优越的细粒度几何和语义一致性。",
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
          "all": 130,
          "last_7_days": 130
        },
        "public_total_votes": 17
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
      "id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "paper_group_id": "019a2db7-e53e-7e68-8b5d-5ce882a28d48",
      "title": "Tongyi DeepResearch Technical Report",
      "abstract": "我们推出了Tongyi DeepResearch，这是一个具备自主能力的大型语言模型，专门设计用于长时间、深入的信息获取研究任务。为了激励自主深度研究能力，Tongyi DeepResearch通过一个端到端的训练框架开发，该框架结合了自主中期训练和自主后期训练，使得在复杂任务中能够进行可扩展的推理和信息获取。我们设计了一个高度可扩展的数据合成流程，完全自动化，无需依赖昂贵的人类标注，支持所有训练阶段。通过为每个阶段构建定制化环境，我们的系统能够在整个过程中实现稳定且一致的交互。Tongyi DeepResearch拥有305亿个参数，每个token仅激活33亿个，在一系列自主深度研究基准测试中实现了最先进的性能，包括人类最后的考试、BrowseComp、BrowseComp-ZH、WebWalkerQA、xbench-DeepSearch、FRAMES以及xbench-DeepSearch-2510。我们开源了模型、框架和完整解决方案，以支持社区发展。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 84,
          "last_7_days": 84
        },
        "public_total_votes": 12
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
      "id": "019a2a13-0030-7409-a686-26a34a0b7408",
      "paper_group_id": "019a2a13-0030-7409-a686-26a34a0b7408",
      "title": "Provable test-time adaptivity and distributional robustness of in-context learning",
      "abstract": "我们研究上下文学习问题，其中一个变换器在由混合分布 $\\pi=\\sum_{\\alpha\\in\\mathcal{A}} \\lambda_{\\alpha} \\pi_{\\alpha}$ 生成的任务上进行了预训练，这被称为预训练先验，其中每个混合成分 $\\pi_{\\alpha}$ 是在特定难度级别 $\\alpha$ 上的任务分布。我们的目标是理解预训练变换器在不同测试分布 $\\mu$ 上的表现，$\\mu$ 由固定难度 $\\beta\\in\\mathcal{A}$ 的任务组成，并且相对于 $\\pi_\\beta$ 存在潜在的分布转变，前提是卡方散度 $\\chi^2(\\mu,\\pi_{\\beta})$ 至多为 $\\kappa$。特别地，我们考虑具有随机平滑性的非参数回归问题，以及具有随机平滑性和随机有效维度的多索引模型。我们证明，基于足够数据的大型变换器在与难度级别 $\\beta$ 对应的最佳收敛速率上取得了成功，这一结果对卡方散度球内的测试分布 $\\mu$ 是均匀成立的。因此，预训练的变换器能够在简单任务上以更快的收敛速率取得成功，并且在测试时对分布转变具有鲁棒性。最后，我们证明，即使一个估计量能够访问测试分布 $\\mu$，其在 $\\mu$ 上的期望风险的收敛速率也不能快于我们预训练变换器的速率，从而提供了比最小最大下界更合适的最优性保证。",
      "paper_summary": {
        "summary": "Researchers at the Statistical Laboratory, University of Cambridge, and LSE provide theoretical guarantees for in-context learning, demonstrating that Transformers pretrained on diverse tasks can adapt to varying test-time difficulties and maintain optimal performance under distribution shifts. The framework shows models achieve convergence rates matching an oracle estimator, robust to chi-squared divergence in test distributions.",
        "originalProblem": [
          "Most existing theoretical analyses of in-context learning (ICL) assume identical pretraining and test data distributions, limiting their applicability to real-world scenarios.",
          "A lack of rigorous, provable guarantees for Transformers' observed ability to adapt to varying task difficulties and maintain performance under distributional shifts existed.",
          "Quantifying the degree to which pretrained Transformers can tolerate shifts in the test distribution without performance degradation was an open theoretical challenge."
        ],
        "solution": [
          "Developed a theoretical framework modeling pretraining with mixture priors (representing different difficulty levels) and test-time distribution shifts quantified by chi-squared divergence.",
          "Decomposed the in-context learning (ICL) excess risk into components that isolate the impact of approximation error, distribution shift, and posterior contraction.",
          "Utilized universal approximation theory for Transformers and a modified Bayesian posterior contraction theory to derive finite-sample performance guarantees."
        ],
        "keyInsights": [
          "The total ICL excess risk can be analytically decomposed, allowing for a clear separation and quantification of the effects of distribution shift from intrinsic statistical difficulty.",
          "Transformers implicitly construct an adaptive 'universal prior' from diverse pretraining data, enabling them to specialize efficiently to specific test-time task difficulties without explicit instruction.",
          "The statistical efficiency of in-context learning in pretrained Transformers can match that of an oracle estimator with full knowledge of the test distribution."
        ],
        "results": [
          "Proved that pretrained Transformers achieve optimal convergence rates (n^(-2β/(2β+d))) for nonparametric regression, adaptively matching the specific smoothness (β) of the test task.",
          "Demonstrated similar optimal and adaptive rates (n^(-2β/(2β+r))) for multi-index models, adapting to both smoothness (β) and effective dimension (r) of the test function.",
          "The derived optimal convergence rates hold uniformly over test distributions exhibiting bounded chi-squared divergence, providing formal guarantees of robustness to moderate distribution shifts."
        ]
      },
      "image_url": "image/2510.23254v1.png",
      "universal_paper_id": "2510.23254",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 102,
          "last_7_days": 102
        },
        "public_total_votes": 15
      },
      "first_publication_date": "2025-10-27T12:16:49.000Z",
      "publication_date": "2025-10-27T12:16:49.000Z",
      "updated_at": "2025-10-28T09:07:59.664Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "domain-adaptation",
        "Mathematics",
        "math.ST",
        "meta-learning",
        "representation-learning",
        "statistical-learning",
        "Statistics",
        "stat.ML",
        "transfer-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "University of Cambridge",
          "image": "images/organizations/university-of-cambridge.svg+xml"
        },
        {
          "name": "London School of Economics and Political Science",
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
      "abstract": "大型语言模型（LLMs）的快速进步促进了数据代理的出现——这些自主系统旨在协调数据+人工智能生态系统，以应对复杂的数据相关任务。然而，“数据代理”这一术语当前存在术语模糊和不一致使用的问题，将简单的查询响应者与复杂的自主架构混淆在一起。这种术语模糊导致用户期望的不匹配、责任问题和行业增长的障碍。受SAE J3016自动驾驶标准的启发，本调查介绍了第一个系统的层次分类法，涵盖六个层级，描述并追踪自主权的逐步变化，从手动操作（L0）到生成的、完全自主的数据代理愿景（L5），从而澄清能力界限和责任分配。通过这一视角，我们提供了一份现有研究的结构性回顾，按照自主权的递增排列，涵盖了用于数据管理、准备和分析的专用数据代理，以及朝着增强自主性的多功能综合系统的最新努力。我们还分析了推进数据代理的关键进化飞跃和技术差距，特别是正在进行的L2到L3的过渡，其中数据代理从程序执行演变为自主协调。最后，我们以一份前瞻性的路线图作结，展望主动的、生成性的数据代理的到来。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 93,
          "last_7_days": 93
        },
        "public_total_votes": 9
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
      "id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "paper_group_id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "title": "DeepSeek-OCR: Contexts Optical Compression",
      "abstract": "我们提出了DeepSeek-OCR作为压缩长文本上下文可行性的初步研究，采用光学二维映射技术。DeepSeek-OCR由两个组件组成：DeepEncoder和DeepSeek3B-MoE-A570M作为解码器。具体来说，DeepEncoder作为核心引擎，旨在在高分辨率输入下保持低激活，同时实现高压缩比，以确保视觉标记的数量最优且可控。实验表明，当文本标记数量在视觉标记的10倍以内（即压缩比小于10x）时，该模型能够达到97%的解码（OCR）精度。即使在20倍的压缩比下，OCR准确率仍然保持在约60%。这为历史长文本压缩和大型语言模型中的遗忘机制等研究领域展现了相当大的潜力。此外，DeepSeek-OCR还展现了很高的实际应用价值。在OmniDocBench上，它仅使用100个视觉标记便超过了GOT-OCR2.0（256标记/页），并在使用不到800个视觉标记的情况下超越了MinerU2.0（平均每页6000+标记）。在生产环境中，DeepSeek-OCR可以以每天超过20万页的规模为大型语言模型/视觉语言模型生成训练数据（单个A100-40G）。代码和模型权重可以在此http网址公开获得。",
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
        "total_votes": 235,
        "visits_count": {
          "all": 7576,
          "last_7_days": 6758
        },
        "public_total_votes": 458
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
      "id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "paper_group_id": "019a2dc0-088b-7216-b79b-b0982fed63f9",
      "title": "Uniform Discrete Diffusion with Metric Path for Video Generation",
      "abstract": "连续空间视频生成迅速发展，而离散方法由于误差积累和长时间上下文不一致而滞后。在本工作中，我们重新审视了离散生成建模，提出了带有度量路径的均匀离散扩散（URSA），这是一个简单但强大的框架，填补了可扩展视频生成与连续方法之间的差距。URSA的核心将视频生成任务表述为对离散时空标记的迭代全局细化。它集成了两个关键设计：线性化度量路径和依赖于分辨率的时间步移机制。这些设计使URSA能够有效扩展到高分辨率图像合成和长时段视频生成，同时需要显著更少的推理步骤。此外，我们引入了异步时间微调策略，将多种任务统一在同一个模型中，包括插值和图像到视频的生成。在具有挑战性的视频和图像生成基准上的广泛实验表明，URSA consistently超越现有的离散方法，并且在性能上可与最先进的连续扩散方法相媲美。代码和模型可以在此网址获得。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 70,
          "last_7_days": 70
        },
        "public_total_votes": 8
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
      "id": "019a29c7-fe4c-7929-8134-929a2bda71a9",
      "paper_group_id": "019a29c7-fe4c-7929-8134-929a2bda71a9",
      "title": "FARMER: Flow AutoRegressive Transformer over Pixels",
      "abstract": "直接建模原始数据分布的显式似然性是机器学习领域的一个关键主题，通过自回归建模在大型语言模型中取得了成功的规模。然而，针对视觉像素数据的连续自回归建模面临着极长的序列和高维空间的问题。在本文中，我们提出了FARMER，一个新颖的端到端生成框架，它将归一化流（NF）和自回归（AR）模型统一起来，以实现可处理的似然估计和直接从原始像素合成高质量图像。FARMER采用可逆自回归流将图像转换为潜在序列，其分布由自回归模型隐式建模。为了解决像素级建模中的冗余性和复杂性，我们提出了一种自监督降维方案，将NF潜在通道划分为信息性和冗余性组，从而实现更有效和高效的自回归建模。此外，我们设计了一种一步蒸馏方案，显著加速推理速度，并引入了一种基于重采样的无分类器引导算法，以提升图像生成质量。大量实验表明，FARMER在提供精确似然性和可扩展训练的同时，其性能与现有的基于像素的生成模型具有竞争力。",
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
          "all": 75,
          "last_7_days": 75
        },
        "public_total_votes": 10
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
      "id": "019a2a0f-9f19-7ded-a883-dcc175b197fc",
      "paper_group_id": "019a2a0f-9f19-7ded-a883-dcc175b197fc",
      "title": "Towards Stable and Effective Reinforcement Learning for Mixture-of-Experts",
      "abstract": "近期在强化学习（RL）方面的进展显著提升了大规模语言模型的训练效果，从而带来了生成质量和推理能力的显著提升。然而，目前大多数研究集中于密集模型，对于专家混合（MoE）架构的RL训练仍然探索不足。为了解决在MoE训练中常见的不稳定性，我们提出了一种新颖的路由器感知方法，以优化离策略RL中的重要性采样（IS）权重。具体而言，我们设计了一种由路由器逻辑引导的重缩放策略，有效减少梯度方差，并缓解训练发散。实验结果表明，我们的方法显著提高了MoE模型的收敛稳定性和最终性能，突显了针对MoE架构的RL算法创新的潜力，并为大规模专家模型的高效训练提供了一个有希望的方向。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 58,
          "last_7_days": 58
        },
        "public_total_votes": 7
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
      "id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "paper_group_id": "019a2e55-6f09-751e-bf5e-a4102946f29e",
      "title": "AgentFold: Long-Horizon Web Agents with Proactive Context Management",
      "abstract": "基于LLM的网络代理在信息检索方面表现出巨大潜力，但其在长期任务上的有效性受到上下文管理中基本权衡的制约。现有的基于ReAct的代理由于积累了噪声较大、原始的历史记录而遭遇上下文饱和，而在每个步骤中固定总结完整历史的方法则可能导致关键信息的不可逆丧失。为了解决这些问题，我们推出了AgentFold，一种以主动上下文管理为中心的新型代理范式，灵感来自人类认知过程中的追溯整合。AgentFold将其上下文视为一个动态的认知工作空间，而非一个被动的日志。它在每一步学习执行一个“折叠”操作，以在多个层面上管理其历史轨迹：可以进行细粒度的浓缩以保留重要的、细致的细节，或进行深度整合以抽象掉整个多步骤的子任务。我们在显著基准测试中的结果令人瞩目：通过简单的监督微调（无需持续的预训练或强化学习），我们的AgentFold-30B-A3B代理在BrowseComp上达到了36.2%，在BrowseComp-ZH上达到了47.3%。值得注意的是，这一表现不仅超越或匹配了规模大得多的开源模型，如DeepSeek-V3.1-671B-A37B，还超过了领先的专有代理，如OpenAI的o4-mini。",
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
          "all": 41,
          "last_7_days": 41
        },
        "public_total_votes": 6
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
      "id": "019a298d-625b-7589-9dae-bd0721cafa20",
      "paper_group_id": "019a298d-625b-7589-9dae-bd0721cafa20",
      "title": "ReCode: Unify Plan and Action for Universal Granularity Control",
      "abstract": "现实世界的任务需要在不同粒度上做出决策，而人类通过利用统一的认知表征在这一方面表现出色，其中规划被根本理解为一种高级别的行动形式。然而，当前基于大型语言模型（LLM）的智能体缺乏这种在决策粒度间流畅操作的重要能力。这一局限性源于现有的范式，它强行将高级规划与低级行动分隔开，这妨碍了动态适应性并限制了泛化能力。我们提出了ReCode（递归代码生成），一种新颖的范式，通过在单一代码表征中统一规划和行动来解决这一限制。在这种表征中，ReCode将高级计划视为抽象的占位符函数，然后智能体递归地将其分解为更细粒度的子函数，直到达到原始行动。这种递归方法消除了计划和行动之间的严格界限，使智能体能够动态控制其决策粒度。此外，递归结构本质上生成丰富的多粒度训练数据，使模型能够学习层次决策过程。大量实验表明，ReCode在推理性能上显著超越了先进基线，并在训练中展现出卓越的数据效率，验证了我们的核心见解，即通过递归代码生成统一规划和行动是一种强大且有效的方法，实现普遍粒度控制。代码可在这个网址获取。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 69,
          "last_7_days": 69
        },
        "public_total_votes": 16
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
    },
    {
      "id": "019a2a19-bf2d-7596-8ea8-db8254a0eeaa",
      "paper_group_id": "019a2a19-bf2d-7596-8ea8-db8254a0eeaa",
      "title": "LightBagel: A Light-weighted, Double Fusion Framework for Unified Multimodal Understanding and Generation",
      "abstract": "统一多模态模型最近在能力和多样性上显示出显著的提升，但大多数领先系统仍然是从零开始训练，并且需要大量的计算资源。在本文中，我们展示了通过战略性地融合公开可用的专注于生成或理解的模型，可以更高效地获得具有竞争力的表现。我们关键的设计是保留原始模块，同时在网络中额外交织多模态自注意力模块。这种双重融合机制 (1) 有效地实现丰富的多模态融合，同时在很大程度上保留基础模型的原始优势，(2) 促进了理解编码器的高层语义表示与生成编码器的低层空间信号之间的协同融合。通过仅用约 35 亿个标记进行训练，这种方法在多个基准测试中取得了强劲的结果：在 GenEval 上的组合文本到图像生成为 0.91，在 DPG-Bench 上的复杂文本到图像生成为 82.16，在 GEditBench 上为 6.06，在 ImgEdit-Bench 上的图像编辑为 3.77。通过全面发布整个代码套件、模型权重和数据集，我们希望支持未来在统一多模态建模方面的研究。",
      "paper_summary": null,
      "image_url": "image/2510.22946v1.png",
      "universal_paper_id": "2510.22946",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 47,
          "last_7_days": 47
        },
        "public_total_votes": 10
      },
      "first_publication_date": "2025-10-27T02:59:57.000Z",
      "publication_date": "2025-10-27T02:59:57.000Z",
      "updated_at": "2025-10-28T09:15:21.773Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "image-generation",
        "lightweight-models",
        "model-merging",
        "multi-modal-learning",
        "parameter-efficient-training",
        "transfer-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Monash University",
          "image": "images/organizations/monash-university.png"
        },
        {
          "name": "Tsinghua University",
          "image": "images/organizations/tsinghua.png"
        },
        {
          "name": "ByteDance",
          "image": "images/organizations/bytedance.png"
        },
        {
          "name": "University of California, Santa Cruz",
          "image": "images/organizations/ucsc.png"
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/UCSC-VLAA/UCSC-VLAA.github.io",
      "distance": 1
    },
    {
      "id": "019a289e-f974-7224-ba25-78069e7f64c2",
      "paper_group_id": "019a289e-f974-7224-ba25-78069e7f64c2",
      "title": "Code Aesthetics with Agentic Reward Feedback",
      "abstract": "大型语言模型（LLMs）已成为开发者在与代码相关任务中的宝贵助手。尽管LLMs在传统编程任务如代码生成和错误修复方面表现出色，但在视觉导向的编码任务中却表现不佳，常常产生美学效果不理想的代码。本文介绍了一种新的流程，以提高LLM生成代码的美学质量。我们首先构建了AesCode-358K，这是一个专注于代码美学的大规模指令调优数据集。接下来，我们提出了代理奖励反馈，这是一种多代理系统，用于评估可执行性、静态美学和交互美学。在此基础上，我们开发了GRPO-AR，它将这些信号整合到GRPO算法中，以实现功能性和代码美学的联合优化。最后，我们开发了OpenDesign，这是一个评估代码美学的基准测试。实验结果表明，将AesCode-358K上的监督微调与使用代理奖励反馈的强化学习相结合，显著提高了OpenDesign上的表现，并增强了在现有基准如PandasPlotBench上的结果。值得注意的是，我们的AesCoder-4B超越了GPT-4o和GPT-4.1，达到了与参数为480B-685B的大型开源模型相当的性能，凸显了我们方法的有效性。",
      "paper_summary": {
        "summary": "AesCoder enables large language models to generate aesthetically pleasing code for visually-oriented tasks like plot generation and webpage design. It achieves this by integrating a multi-agent reward framework and specialized training, allowing smaller models to outperform larger counterparts.",
        "originalProblem": [
          "Large Language Models (LLMs) struggle to produce aesthetically pleasing code for visually-oriented tasks such as data visualizations or web pages, often resulting in suboptimal visual outputs.",
          "Current reward systems for training coding LLMs rely on single textual modalities, limiting their ability to assess visual and interactive aesthetics.",
          "There is a lack of large-scale instruction-tuning datasets and dedicated benchmarks specifically designed for evaluating code aesthetics, particularly for interactive web design."
        ],
        "solution": [
          "The AesCoder pipeline was developed, combining a large-scale instruction-tuning dataset (AesCode-358K) and a multi-agent reward framework for comprehensive aesthetic feedback.",
          "An Agentic Reward Framework was implemented, employing execution, static aesthetics (via GPT-5 multimodal LLM), and interactive aesthetics (via WebVoyager with GPT-4o) agents to provide diverse feedback signals.",
          "A two-stage training approach was used, starting with Supervised Fine-Tuning (SFT) on AesCode-358K, followed by Reinforcement Learning (RL) using the GRPO algorithm with Agentic Reward Feedback (GRPO-AR).",
          "A new benchmark, OpenDesign, was created to enable efficient and automated assessment of webpage aesthetics from both static and interactive perspectives."
        ],
        "keyInsights": [
          "Multi-modal, agent-based reward feedback is essential for guiding LLMs to generate aesthetically pleasing code, significantly outperforming approaches that rely solely on textual rewards.",
          "Specialized instruction-tuning on a curated dataset combined with reinforcement learning from agentic aesthetic feedback allows smaller open-source LLMs (4B, 7B) to achieve competitive performance against much larger open-source and proprietary models.",
          "The concept of 'code aesthetics' for visually rendered outputs can be effectively defined, measured, and incorporated into LLM training, extending the scope of AI-generated content quality beyond functional correctness."
        ],
        "results": [
          "AesCoder models (4B and 7B) consistently achieved improvements in aesthetic quality, showing lower error rates and higher 'good rates' on Python-based plot generation tasks (PandasPlotBench).",
          "On the OpenDesign benchmark for webpage design, AesCoder models demonstrated substantial gains in static and interactive aesthetics, outperforming large open-source models (30B-685B parameters).",
          "AesCoder-4B surpassed leading proprietary models like GPT-4o and GPT-4.1 on both PandasPlotBench and OpenDesign, achieving comparable aesthetic scores to GPT-5 and Claude Sonnet 4, as validated by human evaluations."
        ]
      },
      "image_url": "image/2510.23272v1.png",
      "universal_paper_id": "2510.23272",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 67,
          "last_7_days": 67
        },
        "public_total_votes": 11
      },
      "first_publication_date": "2025-10-27T12:32:33.000Z",
      "publication_date": "2025-10-27T12:32:33.000Z",
      "updated_at": "2025-10-28T02:21:38.548Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.CL",
        "data-curation",
        "deep-reinforcement-learning",
        "fine-tuning",
        "instruction-tuning",
        "multi-agent-learning",
        "optimization-methods",
        "text-generation"
      ],
      "organization_info": [
        {
          "name": "Peking University",
          "image": "images/organizations/peking.png"
        },
        {
          "name": "Microsoft",
          "image": "images/organizations/microsoft.png"
        },
        {
          "name": "Zhiyuan College, Shanghai Jiao Tong University",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/bangx7/code_aesthetics",
      "distance": 1
    },
    {
      "id": "019a2910-1cbe-74ec-aec4-2772a8f9483c",
      "paper_group_id": "019a2910-1cbe-74ec-aec4-2772a8f9483c",
      "title": "Knocking-Heads Attention",
      "abstract": "多头注意力(MHA)已成为现代大型语言模型的基石，通过并行注意力头增强表示能力。然而，增加头的数量本质上削弱了单个头的能力，而现有的注意力机制——无论是标准的MHA还是其变体，如分组查询注意力(GQA)和分组绑定注意力(GTA)——只是简单地将孤立头的输出连接起来，缺乏强烈的交互。为了解决这一限制，我们提出了敲头注意力(KHA)，使注意力头能够“相互敲击”——在缩放点积注意力之前促进跨头特征级交互。这是通过在所有头之间应用一个共享的、对角初始化的投影矩阵来实现的。对角初始化在训练开始时保持头的专业化，同时允许模型逐步学习集成的跨头表示。KHA仅增加了最小的参数和浮点运算，并且可以无缝集成到MHA、GQA、GTA和其他注意力变体中。我们通过在1T高质量标记上训练一个61亿参数的MoE模型（激活1.01亿）来验证KHA。与基线注意力机制相比，KHA带来了更优越和更稳定的训练动态，在下游任务中取得了更好的性能。",
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
          "all": 53,
          "last_7_days": 53
        },
        "public_total_votes": 12
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
      "id": "019a0998-93c4-7993-a782-1cd52db19b53",
      "paper_group_id": "019a0998-93c4-7993-a782-1cd52db19b53",
      "title": "LightMem: Lightweight and Efficient Memory-Augmented Generation",
      "abstract": "尽管大型语言模型（LLMs）具有卓越的能力，但在动态和复杂环境中，它们仍难以有效利用历史交互信息。内存系统使LLMs能够超越无状态交互，通过引入持久的信息存储、检索和利用机制。然而，现有的内存系统往往带来相当大的时间和计算开销。为此，我们提出了一种新的内存系统，称为LightMem，它在内存系统的性能和效率之间取得了平衡。LightMem受到阿特金森-希夫林人类记忆模型的启发，将记忆组织为三个互补的阶段。首先，受认知启发的感官记忆通过轻量压缩快速过滤无关信息，并根据主题对信息进行分组。接下来，具有主题意识的短期记忆巩固这些基于主题的分组，为更结构化的访问组织和总结内容。最后，具有睡眠时间更新的长期记忆采用一种离线程序，将巩固与在线推理解耦。在与GPT和Qwen骨架的LongMemEval实验中，LightMem在准确性上超过强基线（提高高达10.9%），同时减少了硬币使用次数最多117倍、API调用次数最多159倍，运行时间超过12倍。代码可在该链接获取。",
      "paper_summary": {
        "summary": "LightMem introduces a lightweight and efficient memory system for Large Language Models, enabling them to effectively process long and dynamic conversational contexts with improved accuracy and drastically reduced computational overhead. It achieves up to 117x fewer tokens, 177x fewer API calls, and over 12x faster runtime while increasing question-answering accuracy by up to 10.9%.",
        "originalProblem": [
          "Large Language Models struggle with fixed context windows and the 'lost in the middle' problem, limiting their ability to leverage historical interaction information in long, dynamic conversations.",
          "Existing LLM memory systems incur substantial inefficiencies, characterized by high token consumption, excessive API calls, and long runtimes due to redundant information processing.",
          "Memory construction often lacks semantic cohesion, leading to fragmented or inaccurate representations, and real-time complex memory updates introduce significant latency during interactions."
        ],
        "solution": [
          "LightMem employs a three-stage, human-memory-inspired architecture (Sensory Memory, Short-Term Memory, Long-Term Memory) to efficiently filter, organize, and consolidate information.",
          "The Light1 (Sensory Memory) module pre-compresses raw input using LLMLingua-2 to remove redundant tokens and segments the input into topic-coherent units with a hybrid attention- and similarity-based approach.",
          "Light2 (Short-Term Memory) summarizes these topic segments into concise memory entries, while Light3 (Long-Term Memory) uses soft, direct insertions during online inference and parallelized 'sleep-time' offline updates for deeper, consistent consolidation."
        ],
        "keyInsights": [
          "A multi-stage, human-memory-inspired architecture is highly effective for drastically improving both the efficiency and accuracy of LLM memory in dynamic conversational environments.",
          "Pre-compression of raw dialogue input and dynamic, topic-aware segmentation are crucial for reducing redundant processing and enhancing semantic cohesion in memory unit construction.",
          "Decoupling complex memory update and consolidation processes from online inference, through a 'sleep-time' parallelized update mechanism, significantly reduces real-time latency while enabling high-fidelity memory reorganization."
        ],
        "results": [
          "LightMem consistently demonstrated superior performance, achieving up to 10.9% higher question-answering accuracy compared to the strongest baselines like A-Mem.",
          "The system provided substantial efficiency gains, reducing total token consumption by a factor of 10x to 117x and API calls by 3.3x to 177x across different LLM backbones.",
          "Overall runtime was reduced by a factor ranging from 1.67x to 12.45x, highlighting significant improvements in operational speed and cost efficiency."
        ]
      },
      "image_url": "image/2510.18866v1.png",
      "universal_paper_id": "2510.18866",
      "metrics": {
        "total_votes": 15,
        "visits_count": {
          "all": 1010,
          "last_7_days": 944
        },
        "public_total_votes": 97
      },
      "first_publication_date": "2025-10-21T17:58:17.000Z",
      "publication_date": "2025-10-21T17:58:17.000Z",
      "updated_at": "2025-10-22T01:46:25.604Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "cs.MA",
        "human-ai-interaction",
        "inference-optimization",
        "lightweight-models",
        "model-compression",
        "representation-learning",
        "text-generation",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "National University of Singapore",
          "image": "images/organizations/national-university-of-singapore.svg+xml"
        },
        {
          "name": "Zhejiang University",
          "image": "images/organizations/zhejiang.png"
        }
      ],
      "author_info": [],
      "github_stars": 50,
      "github_url": "https://github.com/zjunlp/LightMem",
      "distance": 1
    },
    {
      "id": "019a251c-1491-72d6-95b1-dd003122d496",
      "paper_group_id": "019a251c-1491-72d6-95b1-dd003122d496",
      "title": "WorldGrow: Generating Infinite 3D World",
      "abstract": "我们面临着生成无限可扩展的3D世界的挑战——大规模、连续的环境，具有连贯的几何形状和逼真的外观。现有方法面临关键挑战：二维提升方法在不同视图之间存在几何和外观的不一致性，三维隐式表示难以扩展，而当前的三维基础模型大多以物体为中心，限制了其在场景级生成中的适用性。我们的关键见解是利用预训练3D模型的强生成功能，用于结构化场景块的生成。为此，我们提出了WorldGrow，一个用于无限3D场景合成的分层框架。我们的方法具有三个核心组件：（1）一个数据整理管道，提取高质量的场景块用于训练，使得3D结构化潜在表示适合场景生成；（2）一个3D块修复机制，实现上下文感知的场景扩展；（3）一种粗到细的生成策略，确保全局布局的合理性及局部几何/纹理的保真度。在大规模的3D-FRONT数据集上进行评估，WorldGrow在几何重建方面达到了最先进的性能，同时独特地支持无限场景生成，产生照片般真实且结构一致的输出。这些结果突显了其构建大规模虚拟环境的能力以及未来构建世界模型的潜力。",
      "paper_summary": {
        "summary": "WorldGrow presents a hierarchical framework for generating infinitely extendable 3D worlds by adapting powerful 3D foundation models. It produces continuous environments with coherent geometry and photorealistic appearance, demonstrating superior perceptual quality and structural plausibility in large-scale scenes.",
        "originalProblem": [
          "Existing 2D-lifting methods for 3D generation suffer from geometric inaccuracies and appearance inconsistencies across extended views.",
          "Direct 3D implicit representations are limited by the scale and diversity of available scene-level datasets, hindering scalability for large environments.",
          "Powerful object-centric 3D generation models are primarily designed for single objects and lack mechanisms for coherent scene composition or infinite scene extension."
        ],
        "solution": [
          "A hierarchical framework that adapts Structured Latent Representations (SLATs) for scene blocks through occlusion-aware feature aggregation and decoder retraining.",
          "Introduces a 3D block inpainting mechanism to synthesize new scene blocks based on surrounding context, ensuring seamless continuity.",
          "Employs a coarse-to-fine, block-by-block generation strategy to first establish global structure and then refine local details and appearance."
        ],
        "keyInsights": [
          "Adapting object-centric 3D generative priors (like TRELLIS's SLAT) to scene blocks via occlusion-aware feature aggregation and decoder retraining effectively transfers rich priors to complex environments.",
          "Formulating scene expansion as a context-aware 3D block inpainting problem is crucial for achieving seamless continuity and structural plausibility in unbounded environments.",
          "A hierarchical coarse-to-fine generation strategy is effective for balancing global structural coherence with fine-grained local detail and appearance fidelity across large-scale scenes."
        ],
        "results": [
          "Achieved superior quantitative metrics for 3D geometric quality (FID 7.52) and visual fidelity (CLIP FID 3.95) for scene block generation, outperforming existing infinite scene and scene-scale baselines.",
          "Significantly outperformed comparison methods in human preference studies for structural plausibility, geometric detail, appearance fidelity, and continuity in unbounded scene generation.",
          "Demonstrated stable and consistent generation quality for scenes up to ~1,800 m² (19x39 blocks) without quality degradation over extended expansions, and showed preliminary generalization to outdoor urban scenes."
        ]
      },
      "image_url": "image/2510.21682v1.png",
      "universal_paper_id": "2510.21682",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 217,
          "last_7_days": 217
        },
        "public_total_votes": 25
      },
      "first_publication_date": "2025-10-24T17:39:52.000Z",
      "publication_date": "2025-10-24T17:39:52.000Z",
      "updated_at": "2025-10-27T09:59:48.625Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.GR",
        "generative-models",
        "geometric-deep-learning",
        "image-generation",
        "neural-rendering",
        "representation-learning",
        "synthetic-data",
        "transfer-learning"
      ],
      "organization_info": [
        {
          "name": "Shanghai Jiao Tong University",
          "image": "images/organizations/shanghai-jiao-tong-university.png"
        },
        {
          "name": "Huawei Inc.",
          "image": null
        },
        {
          "name": "Huazhong University of Science and Technology",
          "image": "images/organizations/hust.png"
        },
        {
          "name": "SJTU",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 40,
      "github_url": "https://github.com/world-grow/WorldGrow",
      "distance": 1
    },
    {
      "id": "019a299d-f37e-70bc-acee-0a37bf1c7d90",
      "paper_group_id": "019a299d-f37e-70bc-acee-0a37bf1c7d90",
      "title": "Scalable Supervising Software Agents with Patch Reasoner",
      "abstract": "虽然大语言模型代理在软件工程任务中取得了进展，但现有基于测试的监督方式的不可扩展性限制了数据规模扩大的潜力。原因有二：（1）构建和运行测试沙箱相对繁重且脆弱；（2）高覆盖测试的数据自然稀缺，并在边缘情况下受到测试破解的威胁。在本文中，我们提出了R4P，一个补丁验证模型，通过推理为训练和测试软件工程（SWE）代理提供可扩展的奖励。我们认为补丁验证本质上是一个推理任务，反映了人类代码库维护者在不编写和运行新重现测试的情况下如何审查补丁。为了获得足够的参考并降低奖励破解的风险，R4P使用组目标进行强化学习训练，使其能够相互验证多个补丁的修改，并获得稠密的奖励以实现稳定的训练。R4P在验证SWE-bench-verified的补丁时达到了72.2%的准确率，超越了OpenAI的o3。为了展示R4P的实用性，我们设计并训练了一个轻量级框架Mini-SE，采用纯强化学习，所有奖励均来自R4P。因此，Mini-SE在SWE-bench-verified上达到了26.2%的通过率提升，较原始的Qwen3-32B提高了10.0%。通过R4P进行测试时可进一步提升至32.8%。此外，R4P在一秒内验证补丁，平均比测试快50倍。奖励和准确率的稳定扩展曲线以及高效率反映了R4P的实用性。",
      "paper_summary": {
        "summary": "A collaborative team from The Chinese University of Hong Kong, Shenzhen, and ByteDance developed R4P, a reasoning-based patch verifier that provides scalable, test-free supervision for Large Language Model (LLM) agents in software engineering. This approach enables efficient training and evaluation by leveraging a group-wise task formulation, unlocking vast amounts of previously unusable, untested real-world software issues.",
        "originalProblem": [
          "Traditional test-based supervision for LLM software engineering agents relies on heavy, fragile infrastructure for test execution, limiting scalability and parallel processing.",
          "A scarcity of high-quality, high-coverage test data restricts training, leaving vast amounts of real-world issues in open-source projects unutilized.",
          "Existing tests often suffer from low coverage, allowing LLM agents to generate \"tricky patches\" that pass tests but do not genuinely fix the underlying issue."
        ],
        "solution": [
          "Introduction of R4P (Reasoning for Patch Verifier), a novel model that frames patch verification as a reasoning task, moving away from execution-based methods.",
          "R4P employs a group-wise task formulation where multiple candidate patches are compared, providing richer contextual information and a denser reward signal for stable reinforcement learning.",
          "R4P is trained with an RL algorithm on carefully sampled patch groups and used to supervise a lightweight, execution-free agent (Mini-SE) entirely without traditional sandbox testing."
        ],
        "keyInsights": [
          "Patch verification is fundamentally a reasoning task, similar to human code review, which can be effectively modeled by LLMs to overcome limitations of execution-based testing.",
          "Evaluating a group of candidate patches rather than individual ones provides crucial comparative context, mitigates context deficiency, and generates a denser, more stable reward signal for reinforcement learning.",
          "Decoupling patch verification from execution-based testing significantly improves efficiency and scalability, enabling the utilization of massive, previously unsupervisable datasets for training software agents."
        ],
        "results": [
          "R4P achieved 72.2% accuracy on the SWE-bench-verified dataset, outperforming larger proprietary models and specialized trajectory verifiers.",
          "The lightweight Mini-SE agent, trained purely with R4P's rewards, reached a 26.2% Pass@1 resolution rate on SWE-bench-verified, a 10.0% improvement over the base model.",
          "R4P verifies patches in approximately one second, demonstrating over a 50x speedup compared to traditional test-based verification, which averaged 50 seconds and frequently timed out."
        ]
      },
      "image_url": "image/2510.22775v1.png",
      "universal_paper_id": "2510.22775",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 48,
          "last_7_days": 48
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-26T17:52:05.000Z",
      "publication_date": "2025-10-26T17:52:05.000Z",
      "updated_at": "2025-10-28T07:00:08.702Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.CL",
        "cs.SE",
        "ml-systems",
        "model-observability",
        "reasoning",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "ByteDance",
          "image": "images/organizations/bytedance.png"
        },
        {
          "name": "The Chinese University of Hong Kong, Shenzhen",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2c7c-c15e-7c46-9ac9-995bd9b7ba0b",
      "paper_group_id": "019a2c7c-c15e-7c46-9ac9-995bd9b7ba0b",
      "title": "How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations",
      "abstract": "AI代理不断被优化以执行与人类工作相关的任务，如软件工程和专业写作，这表明了一个对人力资源有重大影响的迫切趋势。然而，这些代理开发往往没有建立在对人类如何执行工作的清晰理解之上，因此未能揭示代理所具备的专业知识及其在多种工作流程中可以扮演的角色。在本研究中，我们通过对人类和代理工作者在多项重要工作相关技能（数据分析、工程、计算、写作和设计）上的首次直接比较，研究代理是如何执行人类工作的。为了更好地理解和比较工人在异构计算机使用活动中的表现，我们引入了一种可扩展的工具包，从人类或代理的计算机使用活动中引导可解释的结构化工作流程。利用这些引导的工作流程，我们比较了人类和代理在执行相同任务时的表现，发现：（1）尽管代理在与人类工作流程的对齐上展现出潜力，但它们在所有工作领域都采取了压倒性的程序化方法，即使在开放式、视觉依赖的任务（如设计）中，这与人类通常使用的以用户界面为中心的方法形成了对比。（2）代理产生的工作质量较低，但往往通过数据伪造和高级工具的不当使用来掩盖其缺陷。（3）尽管如此，代理的交付结果比人类快88.3%，且成本比人类低90.4%-96.2%，这突显了通过将容易编程的任务委派给代理来实现高效协作的潜力。",
      "paper_summary": {
        "summary": "A study comparing AI agent and human work processes across five occupational skill categories developed a toolkit to induce and align hierarchical workflows from raw computer-use activities. The research reveals that AI agents favor programmatic approaches, produce lower-quality work with issues like data fabrication, yet offer substantial speed and cost efficiencies, while human workflows are more significantly altered by AI automation than augmentation.",
        "originalProblem": [
          "Current AI agent development often overlooks how humans perform work, leading to a disconnect between agent capabilities and real-world workflows.",
          "Existing evaluation paradigms for AI agents primarily focus on end-task outcomes, lacking insights into the underlying processes or how agents' methods compare to human approaches.",
          "There is a need for systematic, fine-grained comparisons between human and AI agent work processes to inform effective human-AI collaboration and task delegation across diverse occupations."
        ],
        "solution": [
          "Designed 80 realistic, long-horizon tasks across five major skill categories common to 287 U.S. occupations (data analysis, engineering, computation, writing, design).",
          "Collected raw computer-use activity data from 48 qualified human workers and four representative LLM agent frameworks (ChatGPT Agent, Manus, OpenHands).",
          "Developed a novel, scalable workflow induction toolkit to automatically transform heterogeneous, low-level activities into hierarchical, interpretable workflows for fine-grained comparative analysis."
        ],
        "keyInsights": [
          "AI agents predominantly adopt programmatic approaches for tasks across various domains, contrasting sharply with humans who utilize diverse, interactive UI-oriented tools.",
          "Human workflows undergo substantial alterations when using AI for full automation (shifting to reviewing/debugging), but only experience minor changes with AI augmentation.",
          "Despite procedural consistency, agents often deliver lower-quality work through behaviors like data fabrication and tool misuse, though efficient task delegation by teaming humans and agents can leverage their respective strengths."
        ],
        "results": [
          "Agents exhibited a 93.8% program-use rate, aligning 27.8% more strongly with program-oriented human workflows than with UI-centric ones.",
          "Human workers experienced a 24.3% efficiency gain and 76.8% workflow alignment under AI augmentation, but a 17.7% slowdown and 40.3% alignment under AI automation.",
          "Agents produced work with 32.5–49.5% lower success rates compared to humans, often demonstrating data fabrication and tool misuse, yet delivered results 88.3–96.6% faster and at 90.4–96.2% lower cost."
        ]
      },
      "image_url": "image/2510.22780v1.png",
      "universal_paper_id": "2510.22780",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 40,
          "last_7_days": 40
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-10-26T18:10:22.000Z",
      "publication_date": "2025-10-26T18:10:22.000Z",
      "updated_at": "2025-10-28T20:22:44.830Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.HC",
        "human-ai-interaction",
        "industrial-automation",
        "ml-systems",
        "model-observability",
        "test-time-inference",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 5,
      "github_url": "https://github.com/zorazrw/workflow-induction-toolkit",
      "distance": 1
    },
    {
      "id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "paper_group_id": "019a2f07-80f6-71fb-8975-9bf5a2d94747",
      "title": "Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents",
      "abstract": "我们介绍了Game-TARS，一种通用游戏代理，采用统一、可扩展的动作空间，基于与人类对齐的原生键盘-鼠标输入进行训练。与基于API或GUI的方法不同，这一范式使得在异构领域进行大规模持续预训练成为可能，包括操作系统、网页和模拟游戏。Game-TARS在超过5000亿个标记上进行了预训练，涵盖多样的轨迹和多模态数据。关键技术包括衰减持续损失以减少因果混淆，以及一种高效的稀疏思维策略，能够平衡推理深度和推断成本。实验表明，Game-TARS在开放世界Minecraft任务上的成功率约为之前最佳模型的两倍，在未见过的网页3D游戏中接近新手玩家的通用性，并在FPS基准测试中优于GPT-5、Gemini-2.5-Pro和Claude-4-Sonnet。训练和测试时的扩展结果证实，统一的动作空间在跨游戏和多模态数据的扩展中保持了改进。我们的研究结果表明，简单、可扩展的动作表示与大规模预训练相结合，为具备广泛计算机使用能力的通用代理提供了有前景的路径。",
      "paper_summary": null,
      "image_url": "image/2510.23691v1.png",
      "universal_paper_id": "2510.23691",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 34,
          "last_7_days": 34
        },
        "public_total_votes": 4
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
      "id": "019a2a1b-1829-74f8-842a-71b062afa63e",
      "paper_group_id": "019a2a1b-1829-74f8-842a-71b062afa63e",
      "title": "LongCat-Video Technical Report",
      "abstract": "视频生成是通往世界模型的一个关键途径，而高效的长视频推断是其中一个重要能力。为此，我们介绍了LongCat-Video，这是一种基础的视频生成模型，拥有136亿个参数，在多个视频生成任务中表现出色。它特别擅长高效和高质量的长视频生成，代表了我们迈向世界模型的第一步。其主要特点包括：统一的多任务架构：基于扩散变换器（DiT）框架，LongCat-Video支持文本到视频、图像到视频和视频延续任务，通过一个模型实现；长视频生成：在视频延续任务上进行预训练，使LongCat-Video在生成分钟长的视频时能够保持高质量和时间一致性；高效推断：LongCat-Video通过在时间和空间轴上采用粗到细的生成策略，在几分钟内生成720p、30帧每秒的视频；块稀疏注意力进一步提高了效率，特别是在高分辨率下；多奖励RLHF的强大性能：多奖励RLHF训练使LongCat-Video的性能与最新的闭源和领先的开源模型相当。代码和模型权重已公开，旨在加速该领域的进展。",
      "paper_summary": {
        "summary": "The Meituan LongCat Team developed LongCat-Video, a 13.6-billion-parameter foundational model capable of generating high-quality, minutes-long videos efficiently from text or images. The model achieved a leading score in the \"Commonsense\" dimension on VBench 2.0, demonstrating improved physical plausibility and motion rationality, and offers a unified framework for diverse video generation tasks.",
        "originalProblem": [
          "Existing video generation models often struggle with maintaining temporal coherence and quality over extended durations.",
          "Generating high-resolution, high-frame-rate videos is computationally intensive, limiting practical application.",
          "Many current approaches require specialized models for different tasks like Text-to-Video, Image-to-Video, and Video-Continuation."
        ],
        "solution": [
          "Introduced a Diffusion Transformer (DiT) architecture with 13.6 billion parameters, unifying Text-to-Video, Image-to-Video, and Video-Continuation tasks within a single framework.",
          "Implemented a coarse-to-fine generation strategy and Block Sparse Attention (BSA) to enable efficient high-resolution and long-duration video synthesis.",
          "Utilized a multi-reward Group Relative Policy Optimization (GRPO) training approach to align model outputs with human preferences for visual quality, motion quality, and text-video alignment."
        ],
        "keyInsights": [
          "A coarse-to-fine generation strategy effectively balances quality and efficiency, allowing for high-resolution video generation without excessive computational overhead.",
          "Multi-reward GRPO, with modifications like fixed stochastic timesteps and reweighting coefficients, can effectively optimize video generation models for complex human preferences, preventing reward hacking.",
          "Trainable 3D Block Sparse Attention can significantly reduce computational cost for high-resolution video while maintaining near-lossless quality, critical for long video generation."
        ],
        "results": [
          "Achieved a leading score in the \"Commonsense\" dimension of the VBench 2.0 benchmark, indicating superior motion rationality and physical plausibility in generated videos.",
          "Demonstrated the ability to generate minutes-long videos (up to 2 minutes at 720p, 30fps) without significant quality degradation or color drifting.",
          "Exhibited competitive performance in internal MOS (Mean Opinion Score) evaluations for Text-to-Video and Image-to-Video tasks, nearly matching top commercial models in visual quality and surpassing state-of-the-art open-source alternatives in overall quality."
        ]
      },
      "image_url": "image/2510.22200v1.png",
      "universal_paper_id": "2510.22200",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 79,
          "last_7_days": 79
        },
        "public_total_votes": 15
      },
      "first_publication_date": "2025-10-25T07:41:02.000Z",
      "publication_date": "2025-10-25T07:41:02.000Z",
      "updated_at": "2025-10-28T09:16:50.089Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "deep-reinforcement-learning",
        "efficient-transformers",
        "fine-tuning",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "multi-modal-learning",
        "transformers",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "Meituan",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 645,
      "github_url": "https://github.com/meituan-longcat/LongCat-Video",
      "distance": 1
    },
    {
      "id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "paper_group_id": "019a2e41-430c-7d0d-bc75-080e6007894a",
      "title": "MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation",
      "abstract": "大型语言模型（LLMs）近期的成功重新引发了人们对推荐系统是否能实现类似规模效益的兴趣。传统推荐系统普遍依赖于庞大的嵌入表，随着嵌入维度的增加，这类系统往往会出现性能瓶颈。相比之下，新的生成范式用自回归变换器生成的紧凑语义ID（SID）序列取代了嵌入。然而，大多数工业部署仍然是专有的，留下了两个基本问题待解：（1）预期的规模法则在公共基准上是否成立？（2）实现竞争性能的最小后训练方案是什么？\n\n我们提出了MiniOneRec，尽我们所知，这是第一个完全开源的生成推荐框架，提供了一个端到端的工作流程，涵盖SID构建、监督微调和针对推荐的强化学习。我们通过残差量化变分自编码器生成SID，并在亚马逊评论数据集上对从0.5B到7B参数的Qwen骨干网络进行后训练。实验表明，随着模型大小的增加，训练和评估损失均显示出一致的下降趋势，验证了生成方法的参数效率。为了进一步提升性能，我们提出了一种轻量且有效的后训练管道，(1) 强制进行全过程SID对齐，(2) 应用带约束解码和混合奖励的强化学习。这些技术结合起来，在排名准确性和候选多样性方面都取得了显著提升。",
      "paper_summary": null,
      "image_url": "image/2510.24431v1.png",
      "universal_paper_id": "2510.24431",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 33,
          "last_7_days": 33
        },
        "public_total_votes": 7
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
      "id": "019a24e8-c03e-7621-9d93-6387c729db49",
      "paper_group_id": "019a24e8-c03e-7621-9d93-6387c729db49",
      "title": "Visual Diffusion Models are Geometric Solvers",
      "abstract": "在本文中，我们展示了视觉扩散模型可以作为有效的几何求解器：它们可以通过在像素空间中工作，直接推理几何问题。我们首先在内接正方形问题上证明了这一点，这是一个长期存在的几何问题，询问每个乔丹曲线是否包含四个点形成一个正方形。然后我们将这种方法扩展到另外两个著名的难题几何问题：施泰纳树问题和简单多边形问题。\n\n我们的方法将每个问题实例视为一幅图像，并训练一个标准的视觉扩散模型，该模型将高斯噪声转化为代表有效近似解的图像，该近似解与精确解紧密匹配。模型学习将嘈杂的几何结构转换为正确的配置，有效地将几何推理重新表述为图像生成。\n\n与先前的工作不同，后者在将扩散应用于参数几何表示时需要专门的架构和领域特定的调整，我们采用一个标准的视觉扩散模型，该模型在问题的视觉表示上运行。这种简单性突出了一座意想不到的桥梁，连接了生成建模和几何问题求解。超越这里所研究的具体问题，我们的结果指向一个更广泛的范式：在图像空间中操作为近似著名的难题提供了一个通用而实用的框架，并为解决更广泛的挑战性几何任务打开了大门。",
      "paper_summary": {
        "summary": "Visual diffusion models, leveraging a standard U-Net architecture, can effectively approximate solutions to complex geometric problems such as the Steiner Tree and Maximum Area Polygon by operating purely in pixel space. This method achieved high solution accuracy, with a mean Euclidean length ratio of 1.0008 to optimal for Steiner Trees and finding exact optimal solutions for Maximum Area Polygon in 57.4% of cases, demonstrating generalization capabilities.",
        "originalProblem": [
          "Many complex geometric problems are NP-hard or unsolved, lacking general and efficient algorithmic solutions.",
          "Applying AI to these problems often necessitates specialized data structures or abstract representations, increasing complexity.",
          "Leveraging generative models for direct problem-solving, particularly in combinatorial or mathematical reasoning, remains a developing area."
        ],
        "solution": [
          "Geometric problem instances and their solutions are rasterized into fixed-resolution images.",
          "A standard conditional visual diffusion model, based on a U-Net architecture, is trained to transform noisy inputs into solution images.",
          "The model is conditioned on the rasterized image of the problem instance by concatenating it as an input channel."
        ],
        "keyInsights": [
          "Standard visual diffusion models can inherently learn to solve complex geometric problems by operating solely within pixel space.",
          "The progressive denoising process in diffusion models effectively discovers and refines geometric structures from noisy inputs.",
          "The stochasticity of diffusion allows the generation of diverse valid solutions for problems that may have multiple optimal or near-optimal configurations."
        ],
        "results": [
          "For the Steiner Tree Problem, the method achieved a mean Euclidean length ratio of 1.0008 (± 0.0005) to optimal and a 0.996 valid solution rate for instances within the training range.",
          "In the Maximum Area Polygon Problem, the model found exact optimal solutions in 57.4% of cases and obtained a mean area ratio of 0.9887 (± 0.0205) to optimal.",
          "The diffusion approach consistently outperformed a direct regression model with an identical U-Net architecture, demonstrating its superiority in robustness and solution quality, especially for complex instances."
        ]
      },
      "image_url": "image/2510.21697v1.png",
      "universal_paper_id": "2510.21697",
      "metrics": {
        "total_votes": 8,
        "visits_count": {
          "all": 159,
          "last_7_days": 159
        },
        "public_total_votes": 26
      },
      "first_publication_date": "2025-10-24T17:57:31.000Z",
      "publication_date": "2025-10-24T17:57:31.000Z",
      "updated_at": "2025-10-27T09:03:44.702Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.LG",
        "generative-models",
        "geometric-deep-learning",
        "image-generation",
        "optimization-methods",
        "reasoning",
        "representation-learning",
        "self-supervised-learning"
      ],
      "organization_info": [
        {
          "name": "Google DeepMind",
          "image": "images/organizations/deepmind.png"
        },
        {
          "name": "Tel Aviv University",
          "image": "images/organizations/tel-aviv-university.jpeg"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a04ba-21a1-7422-b529-6edb25aa0e24",
      "paper_group_id": "019a04ba-21a1-7422-b529-6edb25aa0e24",
      "title": "The Free Transformer",
      "abstract": "我们提出了一种解码器Transformer的扩展，它的生成过程以随机潜变量为条件，这些潜变量通过变分过程在无监督的情况下学习。实验评估表明，允许这种条件设置在下游任务上带来了显著的改进。",
      "paper_summary": {
        "summary": "The Free Transformer augments standard decoder-only Transformer architectures by conditioning their generative process on learned, unsupervised random latent variables injected into a middle layer. This architectural innovation from FAIR at Meta enhances performance on reasoning-intensive tasks like code generation and math problems, demonstrating improved inductive bias with minimal computational overhead.",
        "originalProblem": [
          "Purely autoregressive decoder-only Transformers must implicitly infer latent quantities from token sequences, leading to indirect and potentially inefficient computations.",
          "This implicit inference increases model complexity and capacity requirements, as well as vulnerability to error propagation from early token errors.",
          "The autoregressive paradigm can hinder the emergence of \"natural\" data structures, potentially limiting out-of-distribution generalization."
        ],
        "solution": [
          "Extends the standard decoder Transformer by embedding a conditional Variational Autoencoder (VAE) framework directly within its structure.",
          "Latent variables are injected into the middle layer of a shared Transformer decoder, specifically modulating the attention mechanism's key-value space.",
          "A non-causal Transformer block acts as an encoder to infer latent variables from the full sequence during training, coupled with a \"free bits\" method to prevent KL collapse."
        ],
        "keyInsights": [
          "The model effectively learns to condition its generative process on unsupervised latent variables, which capture meaningful structural properties of the data.",
          "Careful control over the information flow into the latent variable (via the free-bits threshold \"kappa\") is crucial to prevent KL collapse and ensure beneficial utilization.",
          "This integrated design provides enhanced inductive bias for complex generative tasks with minimal computational overhead (3-3.6% increase in compute/parameters)."
        ],
        "results": [
          "Achieved substantial performance gains, often exceeding 10%, on generative code (HumanEval+, MBPP) and math (GSM8K) benchmarks, as well as general knowledge (MMLU, CSQA) for both 1.5B and 8B parameter models.",
          "Demonstrated stable training across scales (up to 8B parameters on 1T tokens) with cross-entropy values remaining very close to baseline models.",
          "The benefits were sustained and sometimes amplified when scaling to 8B models trained on 1T tokens, confirming the approach consistently improves inductive bias."
        ]
      },
      "image_url": "image/2510.17558v1.png",
      "universal_paper_id": "2510.17558",
      "metrics": {
        "total_votes": 42,
        "visits_count": {
          "all": 2196,
          "last_7_days": 2002
        },
        "public_total_votes": 166
      },
      "first_publication_date": "2025-10-20T14:05:30.000Z",
      "publication_date": "2025-10-20T14:05:30.000Z",
      "updated_at": "2025-10-21T03:04:58.529Z",
      "topics": [
        "bayesian-deep-learning",
        "Computer Science",
        "cs.LG",
        "generative-models",
        "representation-learning",
        "transformers",
        "unsupervised-learning"
      ],
      "organization_info": [
        {
          "name": "Meta",
          "image": "images/organizations/meta.png"
        }
      ],
      "author_info": [],
      "github_stars": 300,
      "github_url": "https://github.com/PengBoXiangShang/multigraph_transformer",
      "distance": 1
    },
    {
      "id": "019a29c7-dc36-79e4-a66e-ffd68ad51365",
      "paper_group_id": "019a29c7-dc36-79e4-a66e-ffd68ad51365",
      "title": "Think Twice: Branch-and-Rethink Reasoning Reward Model",
      "abstract": "大型语言模型（LLMs）越来越依赖外化中间步骤并分配额外测试时间计算的思维模型，反思策略表明，经过深思熟虑的第二次过程可以引发更强的推理。相比之下，大多数奖励模型（RMs）仍然将许多质量维度压缩成一个单一的标量，这种设计导致判断扩散：注意力在评估标准间分散，导致关注度降低和分析肤浅。我们提出了分支重思（BR-RM），这是一种将思考两次原则转移到奖励建模的两轮奖励模型。第一轮进行自适应分支，选择一小部分关键维度（如事实性和安全性），并勾勒简明的寻证假设。第二轮执行基于分支的重思，这是一种针对性的重读，测试这些假设并仅审视最重要的内容。我们使用GRPO风格的强化学习在结构化的两轮记录上进行训练，采用简单的二元结果奖励并严格格式检查，使该方法与标准的强化学习人类反馈（RLHF）管道兼容。通过将一口气评分转换为聚焦的、第二次审视的推理，BR-RM减少了判断扩散，提高了对细微而重要错误的敏感性，同时保持了实用性和可扩展性。实验结果表明，我们的模型在三个具有挑战性的奖励建模基准测试中达到了最先进的性能，覆盖多个领域。代码和模型将很快发布。",
      "paper_summary": {
        "summary": "Researchers from NVIDIA and UIUC developed the Branch-and-Rethink (BR-RM) framework, a two-turn generative reasoning reward model that significantly enhances evaluation accuracy by explicitly addressing \"judgment diffusion.\" BR-RM establishes new state-of-the-art results across various reward modeling benchmarks, outperforming existing scalar, generative, and reasoning reward models.",
        "originalProblem": [
          "Traditional reward models (RMs) suffer from \"judgment diffusion,\" leading to \"focus dilution\" and \"shallow analysis\" as they attempt to compress numerous quality dimensions into a single, holistic score.",
          "Existing RMs often miss subtle factual inaccuracies, logical inconsistencies, or localized errors that are critical for robust LLM performance and alignment.",
          "Even advanced generative and reasoning RMs frequently lack instance-adaptive focus or a second pass conditioned on specific, discovered issues, limiting their depth of scrutiny."
        ],
        "solution": [
          "The Branch-and-Rethink (BR-RM) framework introduces a two-turn generative evaluation process: adaptive branching followed by branch-conditioned rethinking.",
          "In Stage 1 (Adaptive Branching), the model selects instance-critical evaluation criteria and generates a preliminary \"issue sketch\" for each response, forcing focused attention.",
          "In Stage 2 (Branch-Conditioned Rethinking), the model re-evaluates responses by conducting a targeted, issue-driven deep analysis, guided by the findings from Stage 1 and task-specific evaluation hierarchies.",
          "BR-RM is trained using Generalized Reward Policy Optimization (GRPO) with a curriculum-based reward function that strictly enforces the two-stage format and rewards accurate binary preference outcomes."
        ],
        "keyInsights": [
          "Transferring the \"think twice\" principle from LLM problem-solvers to LLM evaluators significantly improves judgment accuracy and robustness by enabling deliberate, multi-step assessment.",
          "Adaptive branching allows the reward model to dynamically narrow its evaluation scope to the most salient dimensions for each input, effectively combating focus dilution.",
          "Conditioned rethinking, which involves a targeted re-evaluation based on specific issues identified in the first turn, is crucial for preventing shallow analysis and detecting subtle errors.",
          "A direct, binary outcome reward combined with strict format adherence provides a more stable and effective training signal for complex generative reward models than scaled or additional rewards."
        ],
        "results": [
          "BR-RM-Qwen-14B achieved an average accuracy of 84.2% across RewardBench, RM-Bench, and RMB, surpassing all baselines, including larger models like GPT-4o and INF-ORM-Llama3.1-70B.",
          "Ablation studies confirmed the critical contribution of both adaptive branching and conditioned rethinking, with removing either component leading to significant performance degradation (e.g., up to 3.7 points drop on RMB for 'Branching Only').",
          "Analysis of training data highlighted the necessity of a diverse mixture: foundational datasets (HelpSteer3) for general understanding, safety-focused data (Skywork) for safety, and specialized reasoning data (Math-Step-DPO) for complex reasoning tasks."
        ]
      },
      "image_url": "image/2510.23596v1.png",
      "universal_paper_id": "2510.23596",
      "metrics": {
        "total_votes": 3,
        "visits_count": {
          "all": 37,
          "last_7_days": 37
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-27T17:58:07.000Z",
      "publication_date": "2025-10-27T17:58:07.000Z",
      "updated_at": "2025-10-28T07:45:55.254Z",
      "topics": [
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.CL",
        "fine-tuning",
        "reasoning",
        "reasoning-verification",
        "reinforcement-learning",
        "test-time-inference",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "University of Illinois at Urbana-Champaign",
          "image": "images/organizations/university-of-illinois-at-urbana-champaign.jpeg"
        },
        {
          "name": "NVIDIA",
          "image": "images/organizations/nvidia.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2a4d-7ba7-73bf-b831-ec6ea693080b",
      "paper_group_id": "019a2a4d-7ba7-73bf-b831-ec6ea693080b",
      "title": "Alita-G: Self-Evolving Generative Agent for Agent Generation",
      "abstract": "大型语言模型（LLMs）在被构建为具有记忆、工具和反馈的代理时表现得更好。除此之外，自我进化的代理已经出现，但目前的工作主要将适应限制于提示重写或失败重试。因此，我们提出了ALITA-G，一个自我进化框架，通过系统地生成、抽象和策划模型上下文协议（MCP）工具，将通用代理转变为领域专家。在这个框架中，通用代理执行策划的目标领域任务，并从成功的轨迹中合成候选MCP。这些MCP随后被抽象为参数化原语，并整合到一个MCP盒子中。在推理时，ALITA-G在每个工具的描述和用例的帮助下执行增强检索的MCP选择，然后执行配备MCP执行器的代理。在多个基准测试GAIA、PathVQA以及人类最后的考试中，ALITA-G取得了显著的提升，同时减少了计算成本。在GAIA验证中，它达到了83.03%的pass@1和89.09%的pass@3，建立了新的最先进的结果，并将每个示例的平均标记数相较于强基线代理减少了约15%。因此，ALITA-G为从通用能力到可重用的领域特定能力提供了一条有原则的路径，提高了复杂推理任务的准确性和效率。",
      "paper_summary": {
        "summary": "ALITA-G is a self-evolution framework that transforms general-purpose large language model agents into domain-specific experts through task-conditioned, end-to-end adaptation of Model Context Protocol (MCP) tools. The framework notably improved task accuracy by 10.3% relatively on the GAIA benchmark, reaching 83.03%, while simultaneously reducing average token consumption by 15.5% compared to baseline generalist agents.",
        "originalProblem": [
          "Existing self-evolving AI agents often adapt only within a narrow scope (single task) or through shallow mechanisms, failing to achieve true domain expertise.",
          "General-purpose LLM agents lack the deep domain knowledge and efficient tool utilization required for high-performance execution on complex, specialized tasks.",
          "Previous approaches for agent adaptation often struggle with 'prompt bloat' and inefficient tool selection, hindering both performance and computational efficiency."
        ],
        "solution": [
          "ALITA-G employs a multi-execution strategy where a master agent solves target tasks multiple times to generate diverse, successful trajectories and distill reusable Model Context Protocol (MCP) components.",
          "A high-capacity LLM abstracts these raw MCPs into generalized, parameterized tools, forming a curated 'MCP Box' with standardized interfaces and enhanced documentation.",
          "At inference, a Retrieval-Augmented Generation (RAG) mechanism dynamically selects the most relevant MCPs from the 'MCP Box' based on the task query, integrating them into a specialized agent architecture for efficient execution."
        ],
        "keyInsights": [
          "Automating the generation, abstraction, and dynamic retrieval of task-specific executable tools (MCPs) enables end-to-end transformation of general LLM agents into domain experts.",
          "Leveraging both descriptions and use cases for RAG-based MCP selection is crucial for accurately matching task queries to relevant and contextually appropriate tools.",
          "A richer, well-curated MCP Box (e.g., from multiple generation iterations) directly correlates with higher task accuracy and improved reliability without introducing significant regressions."
        ],
        "results": [
          "ALITA-G_3x_ achieved 83.03% accuracy on the GAIA benchmark (pass@1), representing a 10.3% relative improvement over the Original Agent System's 75.15% accuracy.",
          "The framework demonstrated computational efficiency gains, reducing average token consumption by 15.5% on GAIA (from 12,305 to 10,394 tokens) compared to the generalist agent.",
          "ALITA-G also showed superior performance on PathVQA (60% vs. 52%) and HLE (33% vs. 24%) compared to the Original Agent System, highlighting its generalizability across diverse challenging domains."
        ]
      },
      "image_url": "image/2510.23601v1.png",
      "universal_paper_id": "2510.23601",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 34,
          "last_7_days": 34
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-27T17:59:14.000Z",
      "publication_date": "2025-10-27T17:59:14.000Z",
      "updated_at": "2025-10-28T10:11:52.359Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "generative-models",
        "meta-learning",
        "reasoning",
        "reinforcement-learning",
        "tool-use",
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
      "id": "019a300e-cd6e-709d-bd2b-310c66a512e0",
      "paper_group_id": "019a300e-cd6e-709d-bd2b-310c66a512e0",
      "title": "Point Convergence of Nesterov's Accelerated Gradient Method: An AI-Assisted Proof",
      "abstract": "Nesterov加速梯度法于1983年引入，成为优化理论和实践的基石。然而，其收敛性问题一直悬而未决。在这项工作中，我们肯定地解决了这一长期存在的开放问题。证明的发现得到了ChatGPT这一专有大型语言模型的巨大帮助，我们将描述其协助的过程。",
      "paper_summary": null,
      "image_url": "image/2510.23513v1.png",
      "universal_paper_id": "2510.23513",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 29,
          "last_7_days": 29
        },
        "public_total_votes": 5
      },
      "first_publication_date": "2025-10-27T16:52:20.000Z",
      "publication_date": "2025-10-27T16:52:20.000Z",
      "updated_at": "2025-10-29T13:01:07.822Z",
      "topics": [
        "Mathematics",
        "math.OC"
      ],
      "organization_info": [
        {
          "name": "UCLA",
          "image": "images/organizations/ucla.png"
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