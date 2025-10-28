const papersData = {
  "papers": [
    {
      "id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "paper_group_id": "019a23e5-088e-7cbd-828d-79523c0470c9",
      "title": "DeepAgent: A General Reasoning Agent with Scalable Toolsets",
      "abstract": "大型推理模型展现了强大的问题解决能力，但现实世界的任务往往需要外部工具和长期的互动。现有的代理框架通常遵循预定义的工作流程，这限制了自主和全局任务的完成。在本文中，我们介绍了DeepAgent，一个端到端的深度推理代理，能够在单一、连贯的推理过程中执行自主思考、工具发现和行动执行。为了应对长期互动的挑战，特别是由于多次调用工具而导致的上下文长度爆炸和互动历史的积累，我们引入了一种自主记忆折叠机制，将过往互动压缩成结构化的情节记忆、工作记忆和工具记忆，从而减少错误积累，同时保留关键信息。为了有效且稳定地教授通用工具的使用，我们开发了一种端到端的强化学习策略，即ToolPO，该策略利用LLM模拟的API，并应用工具调用优势归因，将细粒度的信用分配给工具调用令牌。我们在八个基准测试上进行了大量实验，包括通用工具使用任务（ToolBench、API-Bank、TMDB、Spotify、ToolHop）和下游应用（ALFWorld、WebShop、GAIA、HLE），结果表明DeepAgent在标记工具和开放集工具检索场景中均持续优于基线方法。本工作朝着开发更通用和更强大的现实世界应用代理迈出了重要的一步。代码和演示可在此HTTPS网址获取。",
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
        "total_votes": 11,
        "visits_count": {
          "all": 343,
          "last_7_days": 343
        },
        "public_total_votes": 35
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
      "abstract": "随着人工智能和机器人研究的快速增长，每年产生超过10,000篇论文，研究人员保持最新信息变得越来越困难。快速演变的趋势、多学科工作的兴起以及探索超出自己专业领域的需求都对这一挑战有所贡献。为了解决这些问题，我们提出了一个通用的流程，能够系统地分析任何研究领域：识别新兴趋势、发现跨领域机会，并提供具体的新研究起点。在这项工作中，我们介绍了“真实深度研究”（Real Deep Research，简称RDR），这是一种适用于人工智能和机器人领域的综合框架，特别关注基础模型和机器人技术的进展。我们还简要将我们的分析扩展到其他科学领域。主要论文详细阐述了RDR流程的构建，而附录则提供了对每个分析主题的广泛结果。我们希望这项工作能为从事人工智能及其他领域研究的研究人员提供启发。",
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
        "total_votes": 20,
        "visits_count": {
          "all": 984,
          "last_7_days": 984
        },
        "public_total_votes": 76
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
      "id": "019a2698-a848-7d12-b559-474738d7624c",
      "paper_group_id": "019a2698-a848-7d12-b559-474738d7624c",
      "title": "Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine",
      "abstract": "近期的研究通过编码代理编辑自身代码库来实现自我改进。它们通过扩展策略生成自我修改的树，这些策略倾向于提高软件工程基准性能，假设这意味着更有前景的后续自我修改。然而，我们发现代理的自我改进潜力（元生产力）与其编码基准性能之间存在不匹配，即元生产力-性能不匹配。受哈克斯利的谱系概念启发，我们提出了一种指标（$\\mathrm{CMP}$），该指标将代理后代的基准性能汇总为其自我改进潜力的一个指示。我们展示，在我们的自我改进编码代理开发环境中，访问真实的 $\\mathrm{CMP}$ 足以模拟哥德尔机器在特定假设下的行为。我们引入了哈克斯利-哥德尔机器（HGM），它通过估计 $\\mathrm{CMP}$ 并将其作为指导，搜索自我修改的树。在SWE-bench Verified 和 Polyglot上，HGM outperform了先前的自我改进编码代理开发方法，同时使用了更少的墙时。最后但并非不重要的是，HGM在其他编码数据集和大型语言模型上显示了强大的迁移能力。通过HGM在SWE-bench Verified上使用GPT-5-mini优化的代理，并在SWE-bench Lite上使用GPT-5进行评估，达到了人类水平的性能，匹配了人类设计编码代理的最佳官方检查结果。我们的代码可在此链接获得。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 179,
          "last_7_days": 179
        },
        "public_total_votes": 15
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
      "id": "019a251c-1491-72d6-95b1-dd003122d496",
      "paper_group_id": "019a251c-1491-72d6-95b1-dd003122d496",
      "title": "WorldGrow: Generating Infinite 3D World",
      "abstract": "我们面临生成可以无限扩展的3D世界的挑战——大型、连续的环境，拥有一致的几何形状和逼真的外观。现有的方法面临关键挑战：2D提升方法在视角间存在几何和外观不一致，3D隐式表示难以扩展，而目前的3D基础模型主要以物体为中心，限制了它们在场景级别生成中的适用性。我们的关键洞察是利用预训练的3D模型中强大的生成先验来进行结构化场景块的生成。为此，我们提出了WorldGrow，一个用于无限3D场景合成的分层框架。我们的方法具有三个核心组件：（1）一个数据整理管道，用于提取高质量的场景块以进行训练，使3D结构化潜在表示适合场景生成；（2）一个3D块修复机制，使得场景扩展具有上下文感知能力；（3）一种粗到细的生成策略，确保全球布局的合理性以及局部几何/纹理的保真度。在大规模3D-FRONT数据集上的评估表明，WorldGrow在几何重建方面达到了领先的性能，同时独特地支持无限场景生成，输出逼真且结构一致的结果。这些结果突显了其构建大规模虚拟环境的能力以及未来世界模型的构建潜力。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 138,
          "last_7_days": 138
        },
        "public_total_votes": 17
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
      "id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "paper_group_id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "title": "DeepSeek-OCR: Contexts Optical Compression",
      "abstract": "我们提出了DeepSeek-OCR，作为对通过光学二维映射压缩长文本的可行性的初步研究。DeepSeek-OCR由两个组件组成：DeepEncoder和DeepSeek3B-MoE-A570M作为解码器。具体来说，DeepEncoder作为核心引擎，旨在在高分辨率输入下保持低激活，同时实现高压缩比，以确保视觉标记的数量既最佳又可管理。实验表明，当文本标记数量在视觉标记的10倍以内（即压缩比小于10倍）时，该模型可以实现97%的解码（OCR）精度。即使在20倍的压缩比下，OCR精度仍保持在约60%。这为历史长文本压缩和大语言模型中的记忆遗忘机制等研究领域展现了相当大的潜力。此外，DeepSeek-OCR还展示了较高的实用价值。在OmniDocBench上，它仅使用100个视觉标记就超越了GOT-OCR2.0（每页256个标记），并在使用不足800个视觉标记的情况下优于MinerU2.0（每页平均6000个以上标记）。在生产中，DeepSeek-OCR能够每天生成超过20万页的LLMs/VLMs训练数据（使用单个A100-40G）。代码和模型权重在此HTTP网址上公开获取。",
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
        "total_votes": 222,
        "visits_count": {
          "all": 6915,
          "last_7_days": 6915
        },
        "public_total_votes": 421
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
      "id": "019a24e8-c03e-7621-9d93-6387c729db49",
      "paper_group_id": "019a24e8-c03e-7621-9d93-6387c729db49",
      "title": "Visual Diffusion Models are Geometric Solvers",
      "abstract": "在本文中，我们展示了视觉扩散模型可以作为有效的几何求解器：它们可以通过在像素空间中工作，直接推理几何问题。我们首先在内切平方问题上证明了这一点，这是一个长期以来在几何学中存在的问题，询问每个乔丹曲线是否包含四个点形成一个正方形。接着，我们将这种方法扩展到另外两个著名的难题：斯坦纳树问题和简单多边形问题。\n\n我们的方法将每个问题实例视为图像，并训练一个标准的视觉扩散模型，该模型将高斯噪声转换为表示有效近似解的图像，该近似解与精确解密切匹配。该模型学习将噪声几何结构转变为正确的配置，有效地将几何推理重新放置为图像生成。\n\n与之前的研究不同，后者在将扩散应用于参数几何表示时需要专门的架构和领域特定的调整，我们采用一个标准的视觉扩散模型，该模型作用于问题的视觉表示。这种简单性突显了生成建模与几何问题求解之间令人惊讶的桥梁。超越此处研究的特定问题，我们的结果指向一个更广泛的范式：在图像空间中操作为近似那些著名的难题提供了一个通用而实用的框架，并为解决更广泛的挑战性几何任务打开了大门。",
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
        "total_votes": 3,
        "visits_count": {
          "all": 94,
          "last_7_days": 94
        },
        "public_total_votes": 15
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
      "id": "019a1759-3897-7864-9e1c-c9fbe9ad8173",
      "paper_group_id": "019a1759-3897-7864-9e1c-c9fbe9ad8173",
      "title": "Thought Communication in Multiagent Collaboration",
      "abstract": "自然语言长期以来促进了人类合作，但其损耗性、模糊性和间接性限制了集体智慧的潜力。虽然机器不受这些限制，但大多数基于大型语言模型的多代理系统仍然仅依赖自然语言，交换标记或其嵌入。为了超越语言，我们引入了一种新的范式——思想沟通，使代理能够像心灵感应一样进行直接的心智对话。为了以原则性的方式揭示这些潜在思想，我们将这一过程形式化为一个一般潜变量模型，其中代理状态由潜在思想的未知函数生成。我们证明，在没有辅助信息的非参数环境中，任何一对代理之间的共享和私有潜在思想都可以被识别。此外，思想共享的全局结构，包括哪些代理共享哪些思想以及这些关系是如何构建的，也可以在理论上得到恢复。在既定理论的指导下，我们开发了一个框架，该框架在沟通之前从所有代理中提取潜在思想，并为每个代理分配相关思想及其共享模式。这种范式自然扩展到所有模态，因为大多数观察数据来自隐藏的生成过程。在合成和现实世界基准上的实验验证了该理论，并展示了思想沟通的协作优势。我们希望这项工作能阐明利用隐藏世界的潜力，因为许多挑战仅通过表层观察无法解决，无论计算或数据规模如何。",
      "paper_summary": {
        "summary": "Researchers from Carnegie Mellon University, Meta AI, and Mohamed bin Zayed University of Artificial Intelligence introduce \"thought communication,\" a method for multi-agent LLMs to exchange latent thoughts directly rather than through natural language. The THOUGHTCOMM framework, backed by nonparametric identifiability theory, achieves an average 19.06% relative improvement in accuracy over state-of-the-art multi-agent finetuning on math reasoning benchmarks, demonstrating greater efficiency and scalability.",
        "originalProblem": [
          "Multi-agent LLM collaboration is fundamentally limited by the sequential, ambiguous, and imprecise nature of natural language communication, leading to inter-agent misalignment and failures.",
          "Existing multi-agent communication methods primarily optimize within the natural language framework, failing to address its inherent constraints for developing truly collective intelligence.",
          "Traditional identifiability theory for latent variable models often relies on strong assumptions or auxiliary information, which are not always applicable for recovering agent \"thoughts\" in a general nonparametric setting."
        ],
        "solution": [
          "A new communication paradigm called \"thought communication\" is proposed, enabling AI agents to interact directly \"mind-to-mind\" by exchanging latent thoughts derived from their internal model states.",
          "A robust theoretical framework, grounded in a latent generative model, establishes nonparametric identifiability for shared and private latent thoughts, as well as their structural organization, under general conditions without auxiliary information.",
          "The THOUGHTCOMM framework implements this by using a sparsity-regularized autoencoder to uncover latent thoughts and an agreement-based reweighting strategy with prefix adaptation to inject personalized latent representations into agents' reasoning."
        ],
        "keyInsights": [
          "Direct communication of latent thoughts, bypassing natural language, can overcome the inherent limitations of current multi-agent LLM collaboration, enabling more precise and efficient inter-agent interaction.",
          "Shared and private latent thoughts, along with their structural organization across agents, can be nonparametrically identified under general, practical conditions without relying on auxiliary information.",
          "A lightweight, modular, and task-agnostic framework can effectively enable \"thought communication\" by extracting latent states and injecting them as prefixes, significantly enhancing collaborative performance and scalability for multi-agent systems."
        ],
        "results": [
          "On challenging math reasoning benchmarks (MATH and GSM8K), THOUGHTCOMM achieved an average relative accuracy improvement of 19.06% over \"Multiagent Finetuning\" (a SOTA baseline) and 67.23% over a single-LLM baseline.",
          "Synthetic experiments validated the core identifiability theory, successfully disentangling and recovering shared and private latent variables with higher R^2 scores compared to baselines without sparsity regularization.",
          "THOUGHTCOMM demonstrates high efficiency and scalability, requiring only a lightweight autoencoder and adapter training (computational cost scales with embedding dimension, not LLM parameters), and maintaining robust performance across various LLMs, debate rounds, and prefix lengths."
        ]
      },
      "image_url": "image/2510.20733v1.png",
      "universal_paper_id": "2510.20733",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 275,
          "last_7_days": 275
        },
        "public_total_votes": 33
      },
      "first_publication_date": "2025-10-23T16:48:02.000Z",
      "publication_date": "2025-10-23T16:48:02.000Z",
      "updated_at": "2025-10-24T17:51:54.519Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "cs.MA",
        "generative-models",
        "multi-agent-learning",
        "reasoning",
        "representation-learning",
        "unsupervised-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a13f2-71bc-722c-932a-fc8ebf018ff5",
      "paper_group_id": "019a13f2-71bc-722c-932a-fc8ebf018ff5",
      "title": "LLM-empowered knowledge graph construction: A survey",
      "abstract": "知识图谱（KGs）长期以来一直作为结构化知识表示和推理的基础设施。随着大型语言模型（LLMs）的出现，KG的构建进入了一个新的范式——从基于规则和统计的流程转向以语言驱动的生成框架。本调查提供了LLM赋能的知识图谱构建的最新进展的全面概述，系统地分析了LLM如何重塑经典的三层管道：本体工程、知识提取和知识融合。\n\n我们首先回顾传统的KG方法，以建立概念基础，然后从两个互补的视角审视新兴的LLM驱动的方法：一种是基于结构、规范化和一致性的架构导向范式，另一种是强调灵活性、适应性和开放发现的无架构范式。在每个阶段，我们综合代表性的框架，分析它们的技术机制，并识别其局限性。\n\n最后，调查概述了关键趋势和未来的研究方向，包括基于KG的LLM推理、智能系统的动态知识记忆以及多模态KG构建。通过这一系统性的回顾，我们旨在澄清LLM与知识图谱之间不断演变的互动，架起符号知识工程与神经语义理解之间的桥梁，推动自适应、可解释和智能知识系统的发展。",
      "paper_summary": {
        "summary": "This survey provides a comprehensive analysis of how Large Language Models (LLMs) are reshaping Knowledge Graph Construction (KGC), detailing their impact across ontology engineering, knowledge extraction, and knowledge fusion. It categorizes current LLM-driven approaches, synthesizes representative frameworks, and identifies future research directions for building more adaptive and generative knowledge systems.",
        "originalProblem": [
          "Traditional Knowledge Graph Construction (KGC) methods faced scalability limitations and high reliance on expert human intervention for schema design and data annotation.",
          "Conventional KGC pipelines were rigid in adapting to new domains and suffered from cumulative error propagation across fragmented stages.",
          "These limitations historically hindered the creation of truly self-evolving, large-scale, and dynamic knowledge graphs."
        ],
        "solution": [
          "Large Language Models (LLMs) are leveraged as \"cognitive engines\" to perform generative knowledge modeling and semantic unification, directly synthesizing structured representations from unstructured text.",
          "LLMs orchestrate complex KGC workflows through instruction-driven orchestration (prompt-based interactions), moving beyond fragmented, rule-driven pipelines.",
          "The survey systematically analyzes LLM integration into the three primary stages of KGC: ontology engineering, knowledge extraction, and knowledge fusion, categorizing approaches into schema-based and schema-free paradigms."
        ],
        "keyInsights": [
          "The KGC landscape is undergoing a fundamental shift from traditional rule-based or statistical methods to unified, adaptive, and generative frameworks powered by LLMs.",
          "LLMs facilitate a move from static schemas to dynamic induction, integrate pipeline modularity into generative unification, and enable a transition from symbolic rigidity to semantic adaptability.",
          "Knowledge Graphs are envisioned as dynamic, cognitive infrastructures that can serve as persistent memory and reasoning substrates for LLM agents, enhancing their logical consistency, causal inference, and interpretability."
        ],
        "results": [
          "LLMs empower ontology engineering through top-down (e.g., CQ-based, natural language-based) and bottom-up (e.g., data-driven schema induction) methods, achieving schema quality comparable to novice human modelers and supporting dynamic schema evolution.",
          "In knowledge extraction, LLMs advance both schema-based methods (evolving from fixed to co-evolving schemas) and schema-free methods (e.g., structured generative extraction, Open Information Extraction), demonstrating broad generalization capabilities.",
          "For knowledge fusion, LLMs enable flexible schema-level and instance-level unification through adaptive reasoning and multi-step prompting, progressing towards autonomous, self-correcting workflows in comprehensive frameworks."
        ]
      },
      "image_url": "image/2510.20345v1.png",
      "universal_paper_id": "2510.20345",
      "metrics": {
        "total_votes": 7,
        "visits_count": {
          "all": 322,
          "last_7_days": 322
        },
        "public_total_votes": 39
      },
      "first_publication_date": "2025-10-23T08:43:28.000Z",
      "publication_date": "2025-10-23T08:43:28.000Z",
      "updated_at": "2025-10-24T02:00:47.292Z",
      "topics": [
        "agentic-frameworks",
        "Computer Science",
        "cs.AI",
        "generative-models",
        "information-extraction",
        "multi-modal-learning",
        "neuro-symbolic-ai",
        "reasoning",
        "text-generation",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Xidian University",
          "image": null
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
      "abstract": "人类通过多感官协同学习抽象概念，一旦形成，这些表征通常可以从单一模态中回忆起来。受到这一原理的启发，我们推出了Concerto，这是一种人类概念学习的极简模拟，用于空间认知，结合了3D内模态自蒸馏与2D-3D跨模态联合嵌入。尽管其简单，Concerto学会了更连贯和富有信息的空间特征，通过零样本可视化得以证明。在3D场景感知的线性 probing 中，其性能分别比独立的 SOTA 2D 和 3D 自监督模型提高了 14.2% 和 4.8%，以及它们的特征级联。通过全面微调，Concerto 在多个场景理解基准上创造了新的 SOTA 结果（例如：在 ScanNet 上达到 80.7% 的 mIoU）。我们还推出了一种针对视频提升的点云空间理解的 Concerto 变体，以及一个将 Concerto 表征线性投影到 CLIP 语言空间的转换器，从而实现开放世界感知。这些结果突显出 Concerto 在空间表征上展现出优越的细粒度几何和语义一致性。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 45,
          "last_7_days": 45
        },
        "public_total_votes": 6
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
      "id": "019a144d-2189-701f-8efb-8bd3b4e59870",
      "paper_group_id": "019a144d-2189-701f-8efb-8bd3b4e59870",
      "title": "From Masks to Worlds: A Hitchhiker's Guide to World Models",
      "abstract": "这不是一份典型的世界模型调查；它是一本为那些希望构建世界的人提供的指南。我们并不打算列举每一篇提到“世界模型”的论文。相反，我们遵循一条清晰的道路：从早期的掩蔽模型，这些模型在不同模态中统一了表示学习；到统一架构，这些架构共享单一范式；再到关闭行动-感知循环的交互生成模型；最后到能够在时间维度上维持一致世界的记忆增强系统。我们绕过松散相关的分支，集中于核心：生成的核心、交互循环和记忆系统。我们展示了这条道路是通往真正世界模型的最有前途的路径。",
      "paper_summary": {
        "summary": "A conceptual framework and evolutionary roadmap are proposed for building \"true world models,\" integrating a generative core, an interactive loop, and a persistent memory system into a five-stage progression. The framework aims to unify fragmented research efforts and define key architectural commitments necessary for creating dynamic, interactive, and self-sustaining computational worlds.",
        "originalProblem": [
          "A lack of clear, unifying consensus on what constitutes a \"true world model\" across diverse AI subfields.",
          "Fragmented research efforts often optimize narrow tasks without a coherent vision for interactive and persistent simulated worlds.",
          "Inconsistent definitions and applications of the term \"world model\" have led to ambiguity."
        ],
        "solution": [
          "Proposes an opinionated \"Hitchhiker's Guide\" that synthesizes disparate research into a coherent, evolutionary roadmap for world models.",
          "Establishes a precise, architectural definition of a true world model, integrating a generative heart, an interactive loop, and a persistent memory system.",
          "Outlines a five-stage progression, from mask-based models to unified architectures, interactive generative models, memory-augmented systems, and finally, \"True World Models.\""
        ],
        "keyInsights": [
          "True world models require the deliberate integration of a generative heart, an interactive loop, and a persistent memory system to achieve dynamic, interactive realities.",
          "The field's progression can be understood through a five-stage evolutionary roadmap, where advancements in one stage lay the foundation for the next.",
          "Achieving true world models hinges on addressing three fundamental challenges: coherence (evaluation), compression (scaling), and alignment (safety)."
        ],
        "results": [
          "Articulated a precise architectural definition of a \"true world model\" comprising three indispensable subsystems: Generative Heart, Interactive Loop, and Memory System.",
          "Established a five-stage evolutionary roadmap guiding the development of world models, categorizing existing and future advancements within each stage.",
          "Identified emergent properties for true world models\n\t\t\t\t\t\t\t\t\t—Persistence, Agency, and Emergence—and defined three frontier challenges that must be overcome (Coherence, Compression, Alignment)."
        ]
      },
      "image_url": "image/2510.20668v1.png",
      "universal_paper_id": "2510.20668",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 210,
          "last_7_days": 210
        },
        "public_total_votes": 30
      },
      "first_publication_date": "2025-10-23T15:46:44.000Z",
      "publication_date": "2025-10-23T15:46:44.000Z",
      "updated_at": "2025-10-24T03:39:50.537Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.LG",
        "generative-models",
        "multi-modal-learning",
        "reinforcement-learning",
        "representation-learning",
        "self-supervised-learning",
        "sequence-modeling"
      ],
      "organization_info": [
        {
          "name": "UCLA",
          "image": "images/organizations/ucla.png"
        },
        {
          "name": "National University of Singapore",
          "image": "images/organizations/national-university-of-singapore.svg+xml"
        },
        {
          "name": "Georgia Tech",
          "image": null
        },
        {
          "name": "UC Merced",
          "image": null
        },
        {
          "name": "MeissonFlow Research",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 9,
      "github_url": "https://github.com/M-E-AGI-Lab/Awesome-World-Models",
      "distance": 1
    },
    {
      "id": "019a13ec-c73a-71df-aa73-6eb0795f4098",
      "paper_group_id": "019a13ec-c73a-71df-aa73-6eb0795f4098",
      "title": "KL-Regularized Reinforcement Learning is Designed to Mode Collapse",
      "abstract": "人们普遍认为，优化逆KL散度会导致“模式寻求”，而优化正KL散度则会导致“覆盖质量”，后者在目标是从多个不同模式中进行抽样时更受欢迎。我们通过数学和实证的方式展示，这种直觉并不一定适用于通过逆/正KL正则化进行强化学习（例如，通常与语言模型一起使用）。相反，逆/正KL的选择决定了由正则化系数参数化的最优目标分布的家族。模式覆盖主要取决于其他因素，如正则化强度以及奖励与参考概率之间的相对尺度。此外，我们表明，常用的设置如低正则化强度和可验证奖励相等，往往指定单模态目标分布，这意味着优化目标在构造上就是非多样的。我们利用这些见解构建了一个简单、可扩展且理论上有正当理由的算法。它对奖励幅度的改动很小，但优化了一个目标分布，使其在所有高质量抽样模式上赋予高概率。在实验中，这一简单的修改可以有效地对大语言模型和化学语言模型进行后训练，提高解决方案的质量和多样性，而不依赖任何外部的多样性信号，并且在使用正逆KL时都能有效，因为直接使用往往失败。",
      "paper_summary": {
        "summary": "Researchers from NYU and EPFL demonstrate that KL-regularized reinforcement learning objectives inherently lead to mode collapse, proving that optimal solutions are often unimodal by design rather than due to optimization failures. They introduce Mode Anchored Reward Augmentation (MARA), a simple algorithm that modifies rewards to foster diverse, high-quality outputs across generative AI applications.",
        "originalProblem": [
          "Diversity collapse (mode collapse) is a pervasive issue in reinforcement learning post-training of foundation models, leading to models producing similar outputs despite multiple valid solutions.",
          "Existing approaches to combat diversity collapse are often heuristic and address symptoms rather than the fundamental theoretical cause.",
          "A lack of rigorous understanding of why diversity collapse occurs inherently within KL-regularized reinforcement learning objectives."
        ],
        "solution": [
          "The authors conducted a rigorous theoretical analysis using variational inference to derive the globally optimal target distributions implicitly defined by KL-regularized RL objectives.",
          "They developed Mode Anchored Reward Augmentation (MARA), a simple algorithm that redefines the reward function to ensure the optimal target distribution places uniformly high mass over all high-quality samples.",
          "The theoretical claims and MARA's efficacy were validated through didactic simulations and experiments on Large Language Models (LLMs) and Chemical Language Models (CLMs)."
        ],
        "keyInsights": [
          "KL-regularized RL objectives, under common settings, are intrinsically designed to yield unimodal optimal policies, indicating that diversity collapse is an inherent outcome rather than an optimization failure.",
          "The choice between reverse and forward KL regularization does not primarily determine mode coverage; instead, the regularization strength (β), and relative magnitudes of rewards and reference probabilities are the dominant factors.",
          "The proposed MARA algorithm promotes multimodality by modifying the reward function to equalize the effective reward-to-reference-probability ratio for all high-quality samples, thus directly shaping the target distribution."
        ],
        "results": [
          "Theoretical analysis revealed that optimal KL-regularized policies exhibit exponential sensitivity to even minor reward differences and merely preserve reference policy ratios for equally rewarded samples, thereby leading to mode collapse.",
          "MARA successfully maintained diversity and quality in LLM post-training, outperforming baselines in tasks like verifiable integer generation and creative question answering by achieving higher entropy and distinctness.",
          "Applied to Chemical Language Models for drug discovery, MARA significantly increased the "
        ]
      },
      "image_url": "image/2510.20817v1.png",
      "universal_paper_id": "2510.20817",
      "metrics": {
        "total_votes": 8,
        "visits_count": {
          "all": 187,
          "last_7_days": 187
        },
        "public_total_votes": 34
      },
      "first_publication_date": "2025-10-23T17:59:40.000Z",
      "publication_date": "2025-10-23T17:59:40.000Z",
      "updated_at": "2025-10-24T01:54:35.962Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "fine-tuning",
        "generative-models",
        "optimization-methods",
        "reinforcement-learning",
        "representation-learning",
        "statistical-learning",
        "text-generation",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "New York University",
          "image": "images/organizations/nyu.png"
        },
        {
          "name": "École Polytechnique Fédérale de Lausanne (EPFL)",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a13e9-d98d-7ae9-855d-24e7a7abad50",
      "paper_group_id": "019a13e9-d98d-7ae9-855d-24e7a7abad50",
      "title": "Collective Communication for 100k+ GPUs",
      "abstract": "大型语言模型（LLMs）规模的不断扩大 necessitates 高效的集体通信框架，特别是在训练负载扩展到数十万块GPU时。传统通信方法在这个规模上面临着显著的吞吐量和延迟限制，阻碍了最先进模型的开发和部署。本文介绍了在Meta开发的NCCLX集体通信框架，旨在优化整个LLM生命周期的性能，从大规模训练的同步需求到推理的低延迟要求。该框架旨在支持超过100,000块GPU的集群上的复杂工作负载，确保可靠的、高吞吐量和低延迟的数据交换。对Llama4模型的实证评估显示了通信效率的显著提升。本研究为下一代LLMs在前所未有的规模下运行提供了强有力的解决方案。",
      "paper_summary": {
        "summary": "Meta's NCCLX introduces a new collective communication framework, building a host-driven, zero-copy, and SM-free communication stack called CTran to support training and deploying next-generation Large Language Models like Llama4 on clusters exceeding 100,000 GPUs. The framework improves Llama4 training step latency by up to 12%, accelerates training startup by up to 11x at 96K GPU scale, and enhances Llama4 Maverick inference decoding latency by 15% to 80% across configurations.",
        "originalProblem": [
          "Existing GPU communication libraries like NVIDIA NCCL face scalability and flexibility limitations when applied to multi-dimensional parallelism of LLMs at 100,000+ GPU scales.",
          "NCCL's host-initiated, kernel-driven, copy-based model incurs overheads from CPU scheduling, GPU-to-CPU synchronizations for dynamic arguments, and GPU resource consumption for internal buffer copies.",
          "NVSHMEM offers device-initiated communication but is constrained by symmetric memory region requirements and limited flexibility within the PyTorch ecosystem."
        ],
        "solution": [
          "Introduced NCCLX, a unified host-driven, zero-copy, and SM-free collective communication framework built on a custom transport layer called CTran, designed for 100,000+ GPUs.",
          "CTran employs host-driven CPU background threads for scheduling, enabling rapid deployment of HPC algorithms, co-design with model algorithms, and direct RDMA operations to bypass GPU kernel involvement.",
          "Leverages modern GPU hardware features for zero-copy data transfers, issuing RDMA operations directly between user buffers, eliminating intermediate FIFO buffer copies and freeing up GPU SMs and HBM bandwidth."
        ],
        "keyInsights": [
          "A host-driven communication framework with zero-copy data transfers can fundamentally overcome the scalability and flexibility limitations of traditional GPU communication libraries for ultra-large-scale LLM workloads.",
          "Co-designing the communication stack with PyTorch's memory management and network hardware is crucial for maximizing efficiency and adaptability in production environments.",
          "Addressing operational challenges like scalable initialization, fault tolerance, and comprehensive performance observability tools are as critical as raw communication throughput for robust large-scale AI system deployment."
        ],
        "results": [
          "NCCLX reduced steady training step latency for Llama4 models by up to 12% and achieved up to 11x faster training startup time at 96K GPU scale compared to baseline NCCL.",
          "For Llama4 Maverick inference, NCCLX demonstrated 15% to 80% improvements in end-to-end decoding latency across various distributed configurations.",
          "Zero-copy CTran point-to-point communication achieved 1.09x to 2.7x speedup for medium message sizes over copy-based NCCL, while internal memory management reduced NCCL GPU memory usage by almost 2x."
        ]
      },
      "image_url": "image/2510.20171v1.png",
      "universal_paper_id": "2510.20171",
      "metrics": {
        "total_votes": 9,
        "visits_count": {
          "all": 296,
          "last_7_days": 296
        },
        "public_total_votes": 41
      },
      "first_publication_date": "2025-10-23T03:32:04.000Z",
      "publication_date": "2025-10-23T03:32:04.000Z",
      "updated_at": "2025-10-24T01:51:24.045Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.DC",
        "cs.NI",
        "distributed-learning",
        "efficient-transformers",
        "hardware-aware-algorithms",
        "inference-optimization",
        "ml-systems",
        "model-deployment-systems",
        "model-serving-infrastructure",
        "optimization-methods",
        "training-orchestration",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Meta",
          "image": "images/organizations/meta.png"
        }
      ],
      "author_info": [],
      "github_stars": 131,
      "github_url": "https://github.com/meta-pytorch/torchcomms",
      "distance": 1
    },
    {
      "id": "019a239f-7e8e-7fba-9349-86ac275849cb",
      "paper_group_id": "019a239f-7e8e-7fba-9349-86ac275849cb",
      "title": "The Universal Landscape of Human Reasoning",
      "abstract": "理解信息如何在人的推理中动态积累和转化，长期以来对认知心理学、哲学和人工智能构成了挑战。现有的理论，从经典逻辑到概率模型，揭示了输出或个体建模的某些方面，但并未提供一个统一的、定量的关于一般人类推理动态的描述。为了解决这个问题，我们引入了信息流追踪（IF-Track），该方法利用大型语言模型（LLMs）作为概率编码器，在每个推理步骤中量化信息熵和增益。通过对多样化任务的细致分析，我们的方法首次成功地在一个单一的度量空间中建模了人类推理行为的普遍特征。我们展示了IF-Track捕捉到的重要推理特征，识别了系统性错误模式，并刻画了个体差异。在应用于先进心理理论的讨论时，我们首次在IF-Track中调和了单过程与双过程理论，发现了人工与人类认知的一致性及LLMs如何重塑人类推理过程。这一方法为理论与测量之间建立了定量的桥梁，提供了关于推理结构的机制性洞见。",
      "paper_summary": null,
      "image_url": "image/2510.21623v1.png",
      "universal_paper_id": "2510.21623",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 47,
          "last_7_days": 47
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-24T16:26:36.000Z",
      "publication_date": "2025-10-24T16:26:36.000Z",
      "updated_at": "2025-10-27T03:04:06.542Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "human-ai-interaction",
        "machine-psychology",
        "reasoning",
        "representation-learning",
        "statistical-learning"
      ],
      "organization_info": [
        {
          "name": "University of Illinois at Urbana-Champaign",
          "image": "images/organizations/university-of-illinois-at-urbana-champaign.jpeg"
        },
        {
          "name": "Shanghai Jiao Tong University",
          "image": "images/organizations/shanghai-jiao-tong-university.png"
        },
        {
          "name": "The Chinese University of Hong Kong",
          "image": "images/organizations/chinesehongkong.png"
        },
        {
          "name": "Central South University",
          "image": null
        },
        {
          "name": "Harbin Institute of Technology",
          "image": null
        },
        {
          "name": "The University of Hong Kong",
          "image": "images/organizations/hku.png"
        },
        {
          "name": "Princeton University",
          "image": "images/organizations/princeton.jpg"
        },
        {
          "name": "ByteDance Seed (China)",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a0ed2-33d2-7e5a-b6cf-f29c444992aa",
      "paper_group_id": "019a0ed2-33d2-7e5a-b6cf-f29c444992aa",
      "title": "Semantic World Models",
      "abstract": "基于世界模型的规划为机器人控制提供了一种强大的范式。传统方法训练一个模型，以当前帧和动作为条件预测未来帧，这个模型可以用于规划。然而，预测未来像素的目标通常与实际的规划目标相矛盾；强大的像素重建并不总是与良好的规划决策相关。本文认为，与其将未来帧重建为像素，不如让世界模型仅仅预测与任务相关的未来语义信息。为了实现这种预测，本文将世界建模视为一个关于未来帧中语义信息的视觉问答问题。这种视角使得可以使用与视觉语言模型相同的工具来进行世界建模。因此，视觉语言模型可以通过对图像-动作-文本数据进行监督微调的过程，训练为“语义”世界模型，从而为决策规划提供支持，同时继承预训练视觉语言模型的许多概括性和鲁棒性特性。本文展示了如何利用这样的语义世界模型在开放式机器人任务中进行策略改进，相较于基于重建的动作条件世界建模的典型范式，带来了显著的概括性提升。网站可在此 HTTPS URL 上访问。",
      "paper_summary": {
        "summary": "Researchers at the University of Washington and Sony AI developed Semantic World Models (SWMs) that redefine world modeling for robotics by predicting task-relevant semantic information through visual question answering (VQA) about future states, rather than reconstructing pixels. This approach, leveraging pre-trained Vision-Language Models, improved success rates on complex tasks by over 50% and demonstrated strong generalization to novel compositions and scenes, outperforming pixel-based world models and offline reinforcement learning methods.",
        "originalProblem": [
          "Traditional pixel-based world models prioritize high-fidelity visual reconstruction, which often fails to capture crucial task-relevant semantic details necessary for effective robotic planning.",
          "The objective of predicting future pixels can conflict with the actual planning objective, leading to models that are visually accurate but semantically unhelpful for decision-making.",
          "Existing methods for incorporating task-relevant information often impose additional assumptions or rely on rewards, limiting their general applicability for open-world robotic control."
        ],
        "solution": [
          "The Semantic World Model (SWM) frames world modeling as a Visual Question Answering (VQA) problem about future states, using natural language to query task-relevant information.",
          "SWM leverages pre-trained Vision-Language Models (VLMs) like PaliGemma, adapting them with an action projection matrix to condition future predictions on proposed actions.",
          "A novel State-Action-Question-Answer (SAQA) dataset is programmatically generated in simulation to train the SWM, and both sampling-based and gradient-based planning strategies are employed."
        ],
        "keyInsights": [
          "Pixel-level reconstruction is often unnecessary and can be detrimental for robotic planning; focusing on semantic prediction directly addresses task-relevant information needs.",
          "Large pre-trained Vision-Language Models provide a powerful foundation for robotic world models, offering strong generalization and semantic understanding from internet-scale data.",
          "SWM can effectively learn from a mixture of optimal and suboptimal data, improving its accuracy and robustness, which is crucial for real-world data collection."
        ],
        "results": [
          "SWM dramatically improved success rates, raising performance on LangTable tasks from 14.4% to 81.6% and OGBench tasks from 45.33% to 76% using gradient-based policy improvement.",
          "The model demonstrated superior generalization to novel object compositions and scene changes, outperforming pixel-based world models and offline reinforcement learning baselines in out-of-distribution scenarios.",
          "Gradient-based planning with SWM was significantly more efficient, executing per action chunk in 1.56 seconds, compared to 676.41 seconds for the pixel-based Action-Conditioned Video Diffusion (AVD) baseline."
        ]
      },
      "image_url": "image/2510.19818v1.png",
      "universal_paper_id": "2510.19818",
      "metrics": {
        "total_votes": 9,
        "visits_count": {
          "all": 346,
          "last_7_days": 346
        },
        "public_total_votes": 46
      },
      "first_publication_date": "2025-10-22T17:53:45.000Z",
      "publication_date": "2025-10-22T17:53:45.000Z",
      "updated_at": "2025-10-23T02:07:28.210Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "cs.RO",
        "fine-tuning",
        "multi-modal-learning",
        "reinforcement-learning",
        "representation-learning",
        "robotic-control",
        "transfer-learning",
        "vision-language-models",
        "visual-qa"
      ],
      "organization_info": [
        {
          "name": "University of Washington",
          "image": "images/organizations/uw.png"
        },
        {
          "name": "Sony AI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 89,
      "github_url": "https://github.com/AgibotTech/EWMBench",
      "distance": 1
    },
    {
      "id": "019a25a3-18de-7a2b-9e88-3e846e88c924",
      "paper_group_id": "019a25a3-18de-7a2b-9e88-3e846e88c924",
      "title": "Sample By Step, Optimize By Chunk: Chunk-Level GRPO For Text-to-Image Generation",
      "abstract": "群体相对策略优化（GRPO）在基于流匹配的文本到图像（T2I）生成中展现出强大的潜力，但面临两个关键限制：不准确的优势归因，以及对生成的时间动态的忽视。在本研究中，我们认为将优化范式从步骤级别转向块级别可以有效缓解这些问题。在此基础上，我们提出了Chunk-GRPO，这是首个基于块级GRPO的T2I生成方法。其核心思想是将连续的步骤归类为一致的“块”，以捕捉流匹配的内在时间动态，并在块级别上优化策略。此外，我们引入了一种可选的加权采样策略，以进一步提升性能。大量实验表明，Chunk-GRPO在偏好对齐和图像质量方面均获得了优异的结果，突显了块级优化在基于GRPO的方法中的潜力。",
      "paper_summary": {
        "summary": "Researchers from Tsinghua University and Kuaishou Technology introduce Chunk-GRPO, an optimization framework for text-to-image generation that refines Group Relative Policy Optimization (GRPO) by optimizing at the chunk level. This method leverages the intrinsic temporal dynamics of flow matching to segment the generation process, achieving superior preference alignment and visual quality compared to existing GRPO approaches.",
        "originalProblem": [
          "Existing GRPO methods for text-to-image (T2I) generation suffer from inaccurate advantage attribution, uniformly applying a single final reward to all timesteps regardless of their individual contributions.",
          "These methods often neglect the prompt-invariant, timestep-dependent temporal dynamics inherent in flow-matching-based T2I processes, treating each step uniformly.",
          "Uniform policy updates across all timesteps can lead to sub-optimal learning signals, hindering fine-grained control for high-quality image generation."
        ],
        "solution": [
          "Chunk-GRPO introduces chunk-level optimization for GRPO, grouping consecutive timesteps and computing importance ratios over chunk likelihoods to provide smoother gradient signals.",
          "Chunk formation is guided by the intrinsic temporal dynamics of flow matching, identified through the relative L1 distance between intermediate latents, ensuring timesteps with similar dynamics are grouped.",
          "An optional weighted sampling strategy biases updates towards high-noise chunks (early timesteps) to accelerate learning in critical generation phases, while acknowledging potential trade-offs with structural stability."
        ],
        "keyInsights": [
          "Optimizing policy updates at a \"chunk\" level, rather than individual timesteps, leads to more accurate advantage attribution and smoother gradient signals in multi-step generative processes.",
          "Flow-matching-based T2I generation exhibits clear, prompt-invariant temporal dynamics that can be leveraged to intelligently segment generation trajectories for more effective reinforcement learning.",
          "While focusing training on high-noise chunks can accelerate preference alignment, it introduces a trade-off with maintaining overall image structure and semantic coherence, suggesting a need for balanced optimization."
        ],
        "results": [
          "Chunk-GRPO consistently achieved higher human preference scores, with HPSv3 scores of 15.373 (vs. 15.080 for Dance-GRPO) and ImageReward scores of 1.149 (vs. 1.141 for Dance-GRPO), demonstrating superior preference alignment.",
          "Qualitative evaluations show images generated by Chunk-GRPO feature stronger lighting contrast, more vivid colors, and finer-grained details compared to baseline methods.",
          "The method achieved an overall score of 0.76 on the rewritten WISE benchmark (vs. 0.75 for Dance-GRPO), showing improved generalization and adherence to diverse textual prompts across categories like Cultural, Time, and Biology."
        ]
      },
      "image_url": "image/2510.21583v1.png",
      "universal_paper_id": "2510.21583",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 41,
          "last_7_days": 41
        },
        "public_total_votes": 6
      },
      "first_publication_date": "2025-10-24T15:50:36.000Z",
      "publication_date": "2025-10-24T15:50:36.000Z",
      "updated_at": "2025-10-27T12:27:17.086Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "deep-reinforcement-learning",
        "generative-models",
        "image-generation",
        "multi-modal-learning",
        "optimization-methods",
        "reinforcement-learning",
        "sequence-modeling",
        "vision-language-models"
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
      "abstract": "现实世界的任务需要在不同粒度上做出决策，而人类擅长通过利用统一的认知表征来实现这一点，其中规划被根本理解为一种高级行动形式。然而，目前基于大型语言模型（LLM）的代理缺乏在决策粒度之间流畅操作的关键能力。这一限制源于现有范式对高层规划和低层行动之间的严格分离，这损害了动态适应能力并限制了泛化。我们提出了ReCode（递归代码生成），这是一种新的范式，通过将规划与行动统一在单一代码表征中来解决这一限制。在这种表征中，ReCode将高层计划视为抽象占位符函数，代理随后递归地将其分解为更细粒度的子函数，直到达到原始动作。这种递归方法 dissolves 了计划与行动之间的严格边界，使代理能够动态控制其决策粒度。此外，递归结构本质上生成了丰富的多粒度训练数据，使模型能够学习分层决策过程。大量实验证明，ReCode在推理性能上显著超越先进基线，并在训练中表现出卓越的数据效率，验证了我们核心的见解：通过递归代码生成统一规划与行动是一种强大且有效的方法，以实现普遍的粒度控制。代码可以在此HTTPS网址获取。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 28,
          "last_7_days": 28
        },
        "public_total_votes": 7
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
      "id": "019a289e-f974-7224-ba25-78069e7f64c2",
      "paper_group_id": "019a289e-f974-7224-ba25-78069e7f64c2",
      "title": "Code Aesthetics with Agentic Reward Feedback",
      "abstract": "大型语言模型（LLMs）已成为开发者在代码相关任务中的宝贵助手。尽管LLMs在传统的编程任务（如代码生成和调试）中表现出色，但它们在以视觉为导向的编码任务中却表现挣扎，往往生成美观度不佳的代码。本文中，我们介绍了一种新管道，以提高LLM生成代码的美学质量。我们首先构建了AesCode-358K，这是一个专注于代码美学的大规模指令调优数据集。接下来，我们提出了代理奖励反馈，这是一种多代理系统，用于评估可执行性、静态美学和交互美学。在此基础上，我们开发了GRPO-AR，将这些信号整合到GRPO算法中，以实现功能性和代码美学的联合优化。最后，我们开发了OpenDesign，这是一个评估代码美学的基准测试。实验结果表明，将AesCode-358K的监督微调与使用代理奖励反馈的强化学习相结合，显著提高了在OpenDesign上的性能，并增强了在现有基准（如PandasPlotBench）上的结果。值得注意的是，我们的AesCoder-4B超过了GPT-4o和GPT-4.1，且在性能上可与480B-685B参数的大型开源模型相媲美，突显了我们方法的有效性。",
      "paper_summary": null,
      "image_url": "image/2510.23272v1.png",
      "universal_paper_id": "2510.23272",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 27,
          "last_7_days": 27
        },
        "public_total_votes": 5
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
      "id": "019a0996-2ae0-7896-b916-33343484c978",
      "paper_group_id": "019a0996-2ae0-7896-b916-33343484c978",
      "title": "A Definition of AGI",
      "abstract": "缺乏对通用人工智能（AGI）的具体定义掩盖了今天专用人工智能与人类水平认知之间的差距。本文引入了一个可量化的框架来解决这个问题，将AGI定义为与受过良好教育的成年人在认知灵活性和熟练程度上相匹配。为了实现这一目标，我们的方法论以卡特尔-霍恩-卡罗尔理论为基础，这一理论是对人类认知的最有经验验证的模型。该框架将一般智力分解为十个核心认知领域，包括推理、记忆和感知，并适应已建立的人类心理测量工具，以评估人工智能系统。应用这一框架揭示了当代模型中高度“锯齿状”的认知特征。尽管在知识密集型领域表现出色，但目前的人工智能系统在基础认知机制上存在关键缺陷，特别是长期记忆存储。因此，AGI得分（例如，GPT-4为27%，GPT-5为57%）具体量化了快速进展以及距AGI之间的 substantial gap。",
      "paper_summary": {
        "summary": "Researchers from the Center for AI Safety, Université de Montréal, Stanford, MIT, and other institutions propose a quantifiable framework for Artificial General Intelligence (AGI), defining it as an AI matching a well-educated adult's cognitive versatility and proficiency. The framework, rooted in the Cattell-Horn-Carroll theory, evaluates AI across ten core cognitive domains, revealing that current models like GPT-4 (27% AGI score) exhibit “jagged” cognitive profiles with significant deficits in areas such as long-term memory storage.",
        "originalProblem": [
          "The term \"AGI\" is ambiguously defined, hindering objective assessment of AI progress and productive scientific discourse.",
          "Existing AGI definitions often rely on task-oriented benchmarks or economic value, failing to capture the holistic breadth of human intelligence and being susceptible to \"capability contortions\" where AI systems leverage narrow strengths.",
          "A lack of a rigorous, quantifiable framework makes it difficult to pinpoint specific strengths and weaknesses of current AI models, impeding targeted research and development efforts."
        ],
        "solution": [
          "A concrete definition of AGI is proposed: \"an AI that can match or exceed the cognitive versatility and proficiency of a well-educated adult,\" emphasizing broad abilities and high skill levels.",
          "The Cattell-Horn-Carroll (CHC) theory of human intelligence is adapted to create a hierarchical, multimodal evaluation framework spanning ten core cognitive domains (e.g., General Knowledge, Reasoning, Long-Term Memory).",
          "The methodology utilizes \"task specifications\" rather than fixed datasets, drawing on psychometric tests and AI benchmarks, to assess underlying cognitive machinery and produce a quantifiable AGI Score (0-100%) and a detailed cognitive profile."
        ],
        "keyInsights": [
          "Grounding AGI evaluation in empirically validated human cognitive theory (CHC) provides a robust and less ambiguous standard compared to purely task-oriented or economic definitions.",
          "Current AI models exhibit \"jagged\" cognitive profiles, demonstrating impressive performance in specialized areas (e.g., General Knowledge, Reading/Writing) but critical foundational deficits (e.g., near-total absence of Long-Term Memory Storage, persistent hallucinations).",
          "The framework exposes \"capability contortions\" where AI systems use workarounds like large context windows or external tools to mimic general intelligence, highlighting that true AGI requires fundamental architectural and conceptual advancements rather than just scaling current approaches."
        ],
        "results": [
          "Applying the framework, GPT-4 achieved an estimated AGI score of 27%, demonstrating strengths in General Knowledge (8%) and Reading and Writing Ability (6%), but near-zero capabilities in On-the-Spot Reasoning and Long-Term Memory Storage (0%).",
          "A projected GPT-5 is estimated at 57% AGI score, showing significant advancements in Reading and Writing, Mathematical Ability (both 10%), and On-the-Spot Reasoning (7%).",
          "Critical bottlenecks persist across both models, with Long-Term Memory Storage remaining at 0% for GPT-5, and Long-Term Memory Retrieval suffering from persistent hallucinations (0% precision), indicating fundamental cognitive machinery for human-like general intelligence is still absent."
        ]
      },
      "image_url": "image/2510.18212v2.png",
      "universal_paper_id": "2510.18212",
      "metrics": {
        "total_votes": 13,
        "visits_count": {
          "all": 641,
          "last_7_days": 641
        },
        "public_total_votes": 66
      },
      "first_publication_date": "2025-10-21T01:28:35.000Z",
      "publication_date": "2025-10-23T18:00:45.000Z",
      "updated_at": "2025-10-22T01:43:47.680Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.LG",
        "machine-psychology",
        "model-interpretation",
        "multi-modal-learning",
        "reasoning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "KAIST",
          "image": "images/organizations/kaist.png"
        },
        {
          "name": "University of Washington",
          "image": "images/organizations/uw.png"
        },
        {
          "name": "University of Toronto",
          "image": "images/organizations/university-of-toronto.jpeg"
        },
        {
          "name": "Carnegie Mellon University",
          "image": "images/organizations/cmu.jpg"
        },
        {
          "name": "Université de Montréal",
          "image": "images/organizations/universit-de-montral.png"
        },
        {
          "name": "New York University",
          "image": "images/organizations/nyu.png"
        },
        {
          "name": "University of Chicago",
          "image": "images/organizations/university-of-chicago.png"
        },
        {
          "name": "UC Berkeley",
          "image": "images/organizations/berkeley.png"
        },
        {
          "name": "University of Oxford",
          "image": "images/organizations/oxford.jpg"
        },
        {
          "name": "Stanford University",
          "image": "images/organizations/stanford.png"
        },
        {
          "name": "University of Michigan",
          "image": "images/organizations/umich.png"
        },
        {
          "name": "Cornell University",
          "image": "images/organizations/cornell.png"
        },
        {
          "name": "Nanyang Technological University",
          "image": "images/organizations/nanyang-technological-university.png"
        },
        {
          "name": "Vector Institute",
          "image": null
        },
        {
          "name": "LG AI Research",
          "image": null
        },
        {
          "name": "MIT",
          "image": "images/organizations/mit.jpg"
        },
        {
          "name": "HKUST",
          "image": "images/organizations/hkust.jpg"
        },
        {
          "name": "University of Tübingen",
          "image": null
        },
        {
          "name": "Hong Kong Baptist University",
          "image": null
        },
        {
          "name": "University of California, Santa Cruz",
          "image": "images/organizations/ucsc.png"
        },
        {
          "name": "Center for AI Safety",
          "image": null
        },
        {
          "name": "Gray Swan AI",
          "image": null
        },
        {
          "name": "Beneficial AI Research",
          "image": null
        },
        {
          "name": "Conjecture",
          "image": null
        },
        {
          "name": "LawZero",
          "image": null
        },
        {
          "name": "University of Wisconsin\n–Madison",
          "image": null
        },
        {
          "name": "Morph Labs",
          "image": null
        },
        {
          "name": "Institute for Applied Psychometrics",
          "image": null
        },
        {
          "name": "CSER",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/vanHeemstraSystems/agile_definition_of_ready",
      "distance": 1
    },
    {
      "id": "019a2a1b-1829-74f8-842a-71b062afa63e",
      "paper_group_id": "019a2a1b-1829-74f8-842a-71b062afa63e",
      "title": "LongCat-Video Technical Report",
      "abstract": "视频生成是通往世界模型的关键路径，其中高效的长视频推理是一个重要能力。为此，我们介绍了LongCat-Video，这是一个基础的视频生成模型，拥有136亿个参数，在多个视频生成任务中表现出色。它特别在高效和高质量的长视频生成方面表现突出，代表了我们朝向世界模型迈出的第一步。主要特点包括：针对多任务的统一架构：基于扩散变换器（DiT）框架，LongCat-Video支持文本到视频、图像到视频和视频续播任务，使用同一个模型；长视频生成：在视频续播任务上的预训练使LongCat-Video能够在生成长达几分钟的视频时保持高质量和时间连贯性；高效推理：LongCat-Video通过沿时间和空间轴采用粗到细的生成策略，在几分钟内生成720p、30fps的视频；块稀疏注意力机制进一步提升了效率，尤其在高分辨率下；与多奖励RLHF结合的强大性能：多奖励RLHF训练使LongCat-Video的性能与最新的闭源和领先的开源模型相当。代码和模型权重公开可用，以加速该领域的进展。",
      "paper_summary": null,
      "image_url": "image/2510.22200v1.png",
      "universal_paper_id": "2510.22200",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 24,
          "last_7_days": 24
        },
        "public_total_votes": 4
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
      "id": "019a13ea-5045-7cc5-afcb-89e288ddde75",
      "paper_group_id": "019a13ea-5045-7cc5-afcb-89e288ddde75",
      "title": "Teaching Language Models to Reason with Tools",
      "abstract": "大型推理模型（LRMs）如OpenAI-o1在自然语言推理方面展现了令人印象深刻的能力。然而，这些模型在处理复杂数学运算时常常表现出低效或不准确的情况。虽然整合计算工具，如代码解释器（CIs），提供了一种有前景的解决方案，但也引入了一个关键挑战：模型内部的概率推理与CI提供的外部确定性知识之间的冲突，这常常导致模型进行无效的思考。为了解决这个问题，我们提出了CoRT（代码优化推理训练），这是一个后训练框架，旨在教会LRMs有效地利用CIs。我们提出了\\emph{Hint-Engineering}，一种新的数据合成策略，在推理路径的最佳点上战略性地注入多样化的提示。这种方法生成了高质量的、代码集成的推理数据，特别针对优化LRM-CI交互而设计。通过这种方法，我们合成了30个高质量样本，以监督微调的方式对1.5B至32B参数的模型进行后训练。CoRT进一步通过使用拒绝采样和强化学习，优化了外部CI使用和内部思考的多轮交错。我们的实验评估证明了CoRT的有效性，在五个具有挑战性的数学推理数据集上，DeepSeek-R1-Distill-Qwen-32B和DeepSeek-R1-Distill-Qwen-1.5B分别取得了绝对提升4\\%和8\\%。此外，CoRT显著提高了效率，相比纯自然语言推理基线，对于32B模型的token使用减少了约30\\%，对于1.5B模型减少了50\\%。模型和代码可在此链接获取。",
      "paper_summary": {
        "summary": "CoRT (Code-Optimized Reasoning Training) is a post-training framework that teaches Large Reasoning Models (LRMs) to efficiently integrate computational tools, specifically Code Interpreters, for complex mathematical problem-solving. Developed by researchers from Alibaba, USTC, and CUHK Shenzhen, the framework achieved substantial improvements in accuracy and reduced token usage by 30-50% across challenging benchmarks, demonstrating a qualitative shift to proactive tool utilization.",
        "originalProblem": [
          "Large Language Models (LLMs) struggle with tasks requiring high numerical precision, complex symbolic manipulation, or exact logical operations in mathematics due to their probabilistic nature.",
          "Existing tool-integrated reasoning approaches often lead to inefficient tool use, characterized by 'unproductive deliberation,' delayed computations, and 'code result distrust,' wasting tokens and hindering performance.",
          "There is a scarcity of high-quality, code-integrated reasoning data for training open-source LLMs, especially since methodologies from advanced proprietary models remain undisclosed."
        ],
        "solution": [
          "CoRT introduces a 'Hint-Engineering' data synthesis strategy, creating high-quality training examples by strategically inserting diverse hints to guide models towards optimal tool use.",
          "A multi-stage post-training framework is employed, including Supervised Fine-Tuning (SFT) and Rejection Fine-Tuning (RFT) to shape model behavior for efficient Code Interpreter (CI) interaction.",
          "Code-integrated Reinforcement Learning (RL) using an adapted GRPO algorithm is applied, featuring a persistent execution environment, output masking, and a dual reward system that incentivizes accuracy and penalizes failed code executions."
        ],
        "keyInsights": [
          "High-quality, human-curated data generated via 'Hint-Engineering' (even a small set of 30 samples) is more effective for teaching nuanced tool-use behavior than larger, less targeted datasets.",
          "Explicitly training models to shift from using Code Interpreters primarily for verification to proactive, direct calculation significantly improves both reasoning efficiency and accuracy.",
          "A specialized RL framework with a persistent execution environment and a balanced accuracy-plus-code-execution reward effectively refines multi-round CI interaction, enabling smaller models to achieve state-of-the-art performance."
        ],
        "results": [
          "CoRT achieved an approximate 8% absolute accuracy gain for 1.5B models (58.3%) over their SFT counterparts and competitive performance (81.3-81.8%) for 32B models on challenging math benchmarks.",
          "The framework dramatically reduced token usage, with 32B models consuming ~30% fewer tokens and 1.5B models consuming ~50% fewer tokens compared to natural language baselines.",
          "Hint-Engineering models qualitatively shifted code usage: from 68.2% verification in Prompt-Hint models to a more efficient 51.1% direct calculation, and demonstrated a strong capacity for out-of-distribution generalization by spontaneously using an unseen RDKit library in 81.3% of chemistry problems."
        ]
      },
      "image_url": "image/2510.20342v1.png",
      "universal_paper_id": "2510.20342",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 120,
          "last_7_days": 120
        },
        "public_total_votes": 20
      },
      "first_publication_date": "2025-10-23T08:41:44.000Z",
      "publication_date": "2025-10-23T08:41:44.000Z",
      "updated_at": "2025-10-24T01:51:54.437Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "data-curation",
        "fine-tuning",
        "inference-optimization",
        "reasoning",
        "reinforcement-learning",
        "tool-use",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Shenzhen Research Institute of Big Data",
          "image": null
        },
        {
          "name": "University of Science and Technology of China",
          "image": "images/organizations/university-of-science-and-technology-of-china.svg+xml"
        },
        {
          "name": "The Chinese University of Hong Kong, Shenzhen",
          "image": null
        },
        {
          "name": "Alibaba Inc.",
          "image": null
        },
        {
          "name": "Shenzhen International Center for Industrial and Applied Mathematics",
          "image": null
        },
        {
          "name": "Shenzhen International Center for Industrial and Applied Mathematics, Shenzhen Research Institute of Big Data",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 53,
      "github_url": "https://github.com/ChengpengLi1003/CoRT",
      "distance": 1
    },
    {
      "id": "019a13f2-1df0-7db3-bc33-6193a1ac3a3f",
      "paper_group_id": "019a13f2-1df0-7db3-bc33-6193a1ac3a3f",
      "title": "AlphaFlow: Understanding and Improving MeanFlow Models",
      "abstract": "MeanFlow 最近崭露头角，成为一种从头训练的强大框架，用于少步生成建模，但其成功尚未完全理解。在本研究中，我们显示出 MeanFlow 目标自然分解为两个部分：轨迹流匹配和轨迹一致性。通过梯度分析，我们发现这些项之间存在强烈的负相关性，导致优化冲突和收敛缓慢。受到这些见解的启发，我们引入了 $\\alpha$-Flow，一个广泛的目标家族，将轨迹流匹配、Shortcut Model 和 MeanFlow 统一在一个公式下。通过采用一种课程策略，从轨迹流匹配平滑地退火到 MeanFlow，$\\alpha$-Flow 解开了冲突目标，并实现了更好的收敛。当在基于 vanilla DiT 骨干网的条件下，从头训练 class-conditional ImageNet-1K 256x256 时，$\\alpha$-Flow 在各个规模和设置中始终优于 MeanFlow。我们最大的 $\\alpha$-Flow-XL/2+ 模型在使用 vanilla DiT 骨干网时实现了新的最先进结果，FID 分数为 2.58 (1-NFE) 和 2.15 (2-NFE)。",
      "paper_summary": {
        "summary": "This research from Snap Inc. and the University of Michigan provides a detailed analysis of MeanFlow's objective function, revealing optimization conflicts, and introduces α-Flow, a unified framework for few-step generative models. The α-Flow framework, coupled with a novel curriculum learning strategy, achieves new state-of-the-art FID scores for from-scratch trained models, reaching 2.58 for 1-NFE and 2.15 for 2-NFE on class-conditional ImageNet-1K 256x256.",
        "originalProblem": [
          "High-fidelity image generation from diffusion models often requires hundreds of iterative steps, leading to slow inference speeds.",
          "Empirically successful few-step generative models like MeanFlow, while efficient, lacked a clear theoretical understanding of their objective functions and optimization dynamics, hindering principled improvements.",
          "MeanFlow's training relies heavily on a computationally expensive 'border-case' flow matching supervision (r=t for 75% of samples), which is counter-intuitive for learning large trajectory leaps."
        ],
        "solution": [
          "The MeanFlow objective was algebraically decomposed into Trajectory Flow Matching (L_TFM) and Trajectory Consistency (L_TCc) components to analyze their interactions.",
          "A generalized family of objectives, α-Flow, was introduced, unifying existing few-step models (trajectory flow matching, Shortcut Models, MeanFlow) under a single formulation parameterized by α.",
          "A three-phase curriculum learning strategy was developed for α-Flow, beginning with α=1 pretraining, smoothly annealing α to a small clamping value, and concluding with MeanFlow fine-tuning."
        ],
        "keyInsights": [
          "Empirical gradient analysis revealed a strong negative correlation (typically below -0.4) between the gradients of L_TFM and L_TCc, indicating an inherent optimization conflict within MeanFlow.",
          "MeanFlow's heavy reliance on 'border-case' flow matching acts as a surrogate for L_TFM, mitigating gradient conflicts with L_TCc but at a significant computational cost.",
          "The α-Flow formulation provides a continuous interpolation between flow matching and consistency objectives, offering a unified perspective on various few-step generative model training paradigms."
        ],
        "results": [
          "α-Flow-XL/2+ achieved new state-of-the-art FID scores for from-scratch trained models on class-conditional ImageNet-1K 256x256, reaching 2.58 for 1-NFE and 2.15 for 2-NFE.",
          "The framework consistently outperformed MeanFlow across model scales (DiT-B/2, DiT-XL/2) and settings, with α-Flow-XL/2 yielding FIDs of 2.95 (1-NFE) and 2.34 (2-NFE) compared to MeanFlow-XL/2's 3.47 (1-NFE) and 2.46 (2-NFE) over 240 epochs.",
          "Ablation studies confirmed the effectiveness of the proposed curriculum learning strategy, demonstrating that longer trajectory flow matching pretraining and smooth α annealing significantly improved performance and training stability."
        ]
      },
      "image_url": "image/2510.20771v1.png",
      "universal_paper_id": "2510.20771",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 137,
          "last_7_days": 137
        },
        "public_total_votes": 22
      },
      "first_publication_date": "2025-10-23T17:45:06.000Z",
      "publication_date": "2025-10-23T17:45:06.000Z",
      "updated_at": "2025-10-24T02:00:25.840Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "cs.LG",
        "generative-models",
        "image-generation",
        "optimization-methods",
        "transformers",
        "unsupervised-learning"
      ],
      "organization_info": [
        {
          "name": "University of Michigan",
          "image": "images/organizations/umich.png"
        },
        {
          "name": "Snap Inc.",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 14,
      "github_url": "https://github.com/snap-research/alphaflow",
      "distance": 1
    },
    {
      "id": "019a13ea-e7c7-71a6-9c21-1387f0c4c523",
      "paper_group_id": "019a13ea-e7c7-71a6-9c21-1387f0c4c523",
      "title": "Compress to Impress: Efficient LLM Adaptation Using a Single Gradient Step on 100 Samples",
      "abstract": "最近，Sharma等人提出了一种称为层选择性秩降低（LASER）的方法，证明了修剪精心选择的LLM权重矩阵的高阶分量可以提高下游准确度——而无需任何基于梯度的微调。然而，LASER对每个矩阵进行的详尽搜索（每次都需要完整数据集的前向传播）使其在快速部署中不具实际可行性。我们证明了这一开销可以去除，并发现：(i) 只需检查一个小的、精心选择的矩阵子集——消除了逐层扫描，(ii) 每个矩阵的奇异值的梯度能够明确指出哪些矩阵需要减少，(iii) 通过允许矩阵行聚集在多个子空间并分别对每个聚类进行分解，从而增大分解搜索空间，可以进一步减少对原始训练数据的过拟合，并将准确率提高多达24.6个百分点，最后，(iv) 我们发现，仅在100个样本上进行评估，而不是完整的训练数据——用于计算指示梯度和测量最终准确率——就足以进一步缩短搜索时间；我们解释说，适应下游任务主要受提示风格的主导，而非数据集规模。因此，我们展示了结合这些发现可以为下游任务提供一种快速且稳健的适应算法。总体而言，通过在100个样本上进行一次梯度步骤和快速扫描顶级候选层及分解技术，我们可以在完全不进行微调的情况下将LLM适应于新数据集。",
      "paper_summary": {
        "summary": "Researchers from MIT CSAIL developed an efficient, training-free method to adapt large language models (LLMs) by intelligently pruning specific weight matrix components. The approach achieved over 50x speedup on GPT-J and 22.2x on RoBERTa, and up to a 24 percentage point accuracy improvement on specific tasks compared to prior training-free techniques, utilizing as few as 100 calibration samples.",
        "originalProblem": [
          "The high computational cost and resource demands of traditional LLM fine-tuning methods, even with parameter-efficient techniques.",
          "The original LASER (LAyer-SElective-Rank reduction) method, while effective, suffered from impracticality due to an exhaustive, computationally expensive search for optimal matrices to prune.",
          "A need for rapid and sample-efficient LLM adaptation suitable for resource-constrained environments or on-device deployment."
        ],
        "solution": [
          "A gradient-guided mechanism identifies which weight matrices are most beneficial to compress using a single backward pass on a small calibration set, eliminating exhaustive search.",
          "Multi-subspace factorization, implemented via simple block splitting, enhances compression effectiveness by addressing the heterogeneous structure within LLM weight matrices.",
          "The method integrates a single gradient step and minimal data evaluation into a robust, training-free pipeline for minute-scale LLM adaptation."
        ],
        "keyInsights": [
          "Singular value gradients derived from a single backward pass can effectively identify weight matrices that benefit most from rank reduction.",
          "Only about 100 labeled examples are sufficient for both guiding the matrix selection and evaluating the performance of compressed models.",
          "Weight matrices are better represented as mixtures of low-dimensional subspaces, making multi-subspace (clustered) SVD more effective for denoising and accuracy gains than a single global SVD."
        ],
        "results": [
          "The method achieved up to a 52.0x speedup on GPT-J and 22.2x on RoBERTa compared to the original LASER, with maintained or improved accuracy.",
          "Demonstrated an average accuracy improvement of 0.95 percentage points for GPT-J models and comparable performance for RoBERTa, with a notable 24 percentage point gain on the BigBench-Epistemic Reasoning task for GPT-J.",
          "Ablation studies confirmed that both gradient-guided selection and 100-sample evaluation contributed independently and significantly to the overall efficiency gains."
        ]
      },
      "image_url": "image/2510.20800v1.png",
      "universal_paper_id": "2510.20800",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 73,
          "last_7_days": 73
        },
        "public_total_votes": 12
      },
      "first_publication_date": "2025-10-23T17:58:01.000Z",
      "publication_date": "2025-10-23T17:58:01.000Z",
      "updated_at": "2025-10-24T01:52:33.223Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "few-shot-learning",
        "fine-tuning",
        "lightweight-models",
        "model-compression",
        "optimization-methods",
        "parameter-efficient-training",
        "transfer-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "MIT",
          "image": "images/organizations/mit.jpg"
        },
        {
          "name": "University of Haifa",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a13f4-17c1-7dc7-ba07-79a1305e2849",
      "paper_group_id": "019a13f4-17c1-7dc7-ba07-79a1305e2849",
      "title": "HoloCine: Holistic Generation of Cinematic Multi-Shot Long Video Narratives",
      "abstract": "最先进的文本到视频模型在生成孤立剪辑方面表现出色，但在创建连贯的多镜头叙事上却有所欠缺，而这正是讲故事的精髓。我们通过HoloCine填补了这一“叙事鸿沟”，该模型整体生成整个场景，以确保从第一镜头到最后一镜头的全球一致性。我们的架构通过窗口交叉注意机制实现精确的导演控制，使文本提示定位到特定镜头，同时稀疏的镜头间自注意模式（镜头内密集但镜头间稀疏）确保了所需的高效性，以实现细微尺度的生成。除了在叙事连贯性上设立新的最先进水平外，HoloCine还展示了显著的新兴能力：对角色和场景的持久记忆，以及对电影技巧的直观理解。我们的研究标志着从剪辑合成到自动化电影制作的关键转变，使端到端的电影创作成为一个切实可行的未来。我们的代码可在此链接获取。",
      "paper_summary": {
        "summary": "HoloCine, developed by Ant Group and HKUST, generates coherent, cinematic multi-shot long video narratives from hierarchical text prompts. The framework introduces architectural innovations to maintain global consistency and directorial control while achieving computational efficiency, outperforming prior text-to-video approaches in narrative fidelity and consistency for minute-scale videos.",
        "originalProblem": [
          "Existing text-to-video models excel at single-shot clips but fail to produce coherent, story-driven multi-shot narratives, creating a \"narrative gap\" in generative AI.",
          "Decoupled video generation methods suffer from consistency drift and error accumulation of visual attributes across extended sequences.",
          "Holistic multi-shot generation methods face prohibitive computational costs for longer videos due to quadratic scaling of self-attention and struggle with precise per-shot control as instructions become diluted."
        ],
        "solution": [
          "A comprehensive data pipeline curates and annotates a large-scale multi-shot video dataset with a two-tier hierarchical prompt structure (global and per-shot instructions).",
          "A holistic multi-shot generation architecture processes latent representations for all shots simultaneously within a Diffusion Transformer (DiT) backbone to inherently promote long-range consistency.",
          "Window Cross-Attention is introduced to localize prompt conditioning, allowing specific per-shot instructions to precisely control corresponding visual segments, and Sparse Inter-Shot Self-Attention reduces computational complexity by performing dense attention within shots and sparse attention between shots using summary tokens."
        ],
        "keyInsights": [
          "Holistic generation is critical for maintaining global consistency across characters, environments, and styles throughout multi-shot narratives.",
          "Achieving precise directorial control within a holistic framework is possible through localized prompt conditioning mechanisms, such as Window Cross-Attention.",
          "Computational efficiency for minute-scale multi-shot videos can be achieved by strategically sparsifying self-attention between shots while preserving essential inter-shot information flow for narrative continuity."
        ],
        "results": [
          "HoloCine establishes a new state-of-the-art on a custom benchmark, significantly outperforming prior models (e.g., Wan2.2, StoryDiffusion+Wan2.2) in transition control, inter-shot consistency, intra-shot consistency, and semantic fidelity.",
          "Ablation studies confirm Window Cross-Attention is critical for executing precise shot cuts and following per-shot instructions, and Sparse Inter-Shot Self-Attention offers comparable generative quality to full attention with vastly improved efficiency.",
          "The model exhibits emergent capabilities, including persistent memory for objects and characters across long, complex sequences, and nuanced control over cinematic language, accurately interpreting commands for shot scale, camera angle, and movement."
        ]
      },
      "image_url": "image/2510.20822v1.png",
      "universal_paper_id": "2510.20822",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 106,
          "last_7_days": 106
        },
        "public_total_votes": 18
      },
      "first_publication_date": "2025-10-23T17:59:59.000Z",
      "publication_date": "2025-10-23T17:59:59.000Z",
      "updated_at": "2025-10-24T02:02:35.329Z",
      "topics": [
        "Computer Science",
        "cs.CV"
      ],
      "organization_info": [
        {
          "name": "CUHK",
          "image": "images/organizations/chinesehongkong.png"
        },
        {
          "name": "Ant Group",
          "image": null
        },
        {
          "name": "NTU",
          "image": null
        },
        {
          "name": "HKUST",
          "image": "images/organizations/hkust.jpg"
        },
        {
          "name": "ZJU",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 126,
      "github_url": "https://github.com/yihao-meng/HoloCine",
      "distance": 1
    },
    {
      "id": "019a2a13-0030-7409-a686-26a34a0b7408",
      "paper_group_id": "019a2a13-0030-7409-a686-26a34a0b7408",
      "title": "Provable test-time adaptivity and distributional robustness of in-context learning",
      "abstract": "我们研究了上下文学习问题，其中一个 Transformer 在源自混合分布 $\\pi=\\sum_{\\alpha\\in\\mathcal{A}} \\lambda_{\\alpha} \\pi_{\\alpha}$ 的任务上进行了预训练，这种混合分布称为预训练先验，其中每个混合成分 $\\pi_{\\alpha}$ 是一个特定难度级别的任务分布，索引由 $\\alpha$ 给出。我们的目标是理解当预训练的 Transformer 在不同的测试分布 $\\mu$ 上进行评估时的性能，该测试分布包含固定难度 $\\beta\\in\\mathcal{A}$ 的任务，并且相对于 $\\pi_\\beta$ 可能存在分布偏移，要求卡方散度 $\\chi^2(\\mu,\\pi_{\\beta})$ 至多为 $\\kappa$。具体来说，我们考虑具有随机光滑性的非参数回归问题，以及具有随机光滑性和随机有效维度的多指标模型。我们证明，基于充足数据预训练的大型 Transformer 能够以对应于难度级别 $\\beta$ 的最优收敛速率在随机光滑性的卡方散度球内的测试分布 $\\mu$ 上均匀实现。因此，预训练的 Transformer 能够在更容易的任务上实现更快的收敛速率，并且对测试时的分布偏移具有鲁棒性。最后，我们证明，即使一个估计量能够获取测试分布 $\\mu$，其在 $\\mu$ 上的期望风险的收敛速率也不能超过我们预训练 Transformers 的收敛速率，从而提供了比极小极大下界更合适的最优性保证。",
      "paper_summary": null,
      "image_url": "image/2510.23254v1.png",
      "universal_paper_id": "2510.23254",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 20,
          "last_7_days": 20
        },
        "public_total_votes": 5
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
      "id": "019a290e-1560-7edf-87e7-25f40c709259",
      "paper_group_id": "019a290e-1560-7edf-87e7-25f40c709259",
      "title": "ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models",
      "abstract": "循环神经网络（RNN）为序列建模奠定了基础，但其内在的顺序特性限制了并行计算，造成了扩展的基本障碍。这导致了可以并行化的架构（如变压器）以及最近的状态空间模型（SSM）的主导地位。尽管SSM通过结构化的线性递归实现了有效的并行化，但这种线性约束限制了它们的表现能力，并排除了对复杂非线性序列依赖关系的建模。为了解决这个问题，我们提出了ParaRNN，这是一个打破非线性RNN序列并行化障碍的框架。基于之前的研究，我们将非线性递归关系的序列表述为一个单一的方程组，并使用结合了自定义并行归约的牛顿迭代进行并行求解。我们的实现比简单的顺序应用快了多达665倍，使得在前所未有的规模上训练非线性RNN成为可能。为了展示这一点，我们将ParaRNN应用于LSTM和GRU架构的改编，成功训练了7B参数的模型，其困惑度与类似规模的变压器和Mamba2架构相当。为了加速高效序列建模的研究，我们将ParaRNN代码库发布为一个开源框架，用于非线性RNN的自动训练并行化，使研究人员和从业者能够探索新的非线性RNN模型。",
      "paper_summary": {
        "summary": "Apple researchers introduced ParaRNN, a framework that enables the efficient parallel training of nonlinear Recurrent Neural Networks (RNNs) for large language models, overcoming the historical scalability barrier of sequential RNNs. The method leverages Newton's method combined with parallel scan operations, achieving significant speedups and training 7B parameter nonlinear RNNs with competitive language modeling performance.",
        "originalProblem": [
          "Traditional Recurrent Neural Networks (RNNs) are inherently sequential, making their training computationally prohibitive and unscalable for large language models (LLMs).",
          "Recent State Space Models (SSMs) like Mamba achieve parallelization by enforcing a strict linearity constraint, which limits their expressive power and ability to capture complex nonlinear sequence dependencies.",
          "The broader class of nonlinear RNNs, despite potential expressive advantages and inference efficiency, has been largely unexplorable for large-scale training due to parallelization challenges."
        ],
        "solution": [
          "The core problem of applying a nonlinear RNN is reframed as solving a system of nonlinear equations, which is then efficiently solved using Newton's method.",
          "At each Newton iteration, the resulting linear system (a block bi-diagonal system) is solved in parallel using a generalized prefix sum (parallel scan) operation, achieving logarithmic time complexity for parallel computation.",
          "Jacobian structures of adapted GRU and LSTM cells are simplified to be diagonal or block-diagonal, reducing memory ($O(d_h)$) and computational costs ($O(d_h)$) during parallel reduction, making the approach practical."
        ],
        "keyInsights": [
          "It is possible to efficiently parallelize the training of nonlinear RNNs for large-scale models by reformulating the recurrence as a system of nonlinear equations solvable via Newton's method and parallel scan.",
          "Nonlinearities provide crucial expressive power, enabling ParaRNN models to achieve 100% accuracy on synthetic tasks (e.g., k-hop, Parity) where linear SSMs and even Transformers struggle.",
          "Careful design of RNN cells to yield sparse Jacobians (e.g., diagonal) is key to making Newton's method computationally tractable for parallel reduction at scale, without sacrificing overall model performance significantly."
        ],
        "results": [
          "ParaRNN achieved speedups of up to 665x over naive sequential RNN application and demonstrated competitive training runtimes compared to Mamba, with fully-fused CUDA kernels providing up to 2.6x speedup over Mamba at sequence length L=2^9.",
          "The framework successfully enabled the training of classical nonlinear RNN models (ParaGRU and ParaLSTM) at an unprecedented scale of 7 billion parameters.",
          "ParaRNN models attained perplexity values (ParaLSTM: 9.16, ParaGRU: 9.19) competitive with similarly-sized Transformers (9.55) and Mamba2 (8.62) for 7B parameter models, and consistently outperformed a DCLM Transformer baseline on various downstream tasks."
        ]
      },
      "image_url": "image/2510.21450v1.png",
      "universal_paper_id": "2510.21450",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 37,
          "last_7_days": 37
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-24T13:28:33.000Z",
      "publication_date": "2025-10-24T13:28:33.000Z",
      "updated_at": "2025-10-28T04:23:00.192Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "distributed-learning",
        "generative-models",
        "ml-systems",
        "optimization-methods",
        "sequence-modeling",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Apple",
          "image": "images/organizations/apple.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a27c9-3955-73ea-9e85-8c43a419dd61",
      "paper_group_id": "019a27c9-3955-73ea-9e85-8c43a419dd61",
      "title": "Self-diffusion for Solving Inverse Problems",
      "abstract": "我们提出了一种自我扩散的新框架，用于解决逆问题，而无需依赖于预训练的生成模型。传统的基于扩散的方法需要在干净数据集上训练一个模型，以学习逆转正向噪声过程。然后使用该模型来采样干净的解决方案——从贝叶斯的角度看，这相当于后验采样——这些解决方案与特定任务下观察到的数据一致。相比之下，自我扩散引入了一种自我包含的迭代过程，该过程在加噪和去噪步骤之间交替进行，以逐步精炼对解决方案的估计。在自我扩散的每一步中，噪声被添加到当前估计中，一个自我去噪器，即一个从头随机初始化的未经训练的卷积网络，通过数据保真度损失被连续训练若干次，以根据嘈杂估计预测解决方案。从本质上讲，自我扩散利用了神经网络的光谱偏差，并通过一个调度的噪声过程对其进行调节。该方法不依赖于预训练的评分函数或外部去噪器，仍然对任意的正向算子和嘈杂观测保持自适应，使其具有高度的灵活性和广泛的适用性。我们在多种线性逆问题上展示了我们方法的有效性，表明自我扩散的表现与其他方法相比具有竞争力或优越性。",
      "paper_summary": null,
      "image_url": "image/2510.21417v1.png",
      "universal_paper_id": "2510.21417",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 32,
          "last_7_days": 32
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-24T12:57:22.000Z",
      "publication_date": "2025-10-24T12:57:22.000Z",
      "updated_at": "2025-10-27T22:28:10.197Z",
      "topics": [
        "Computer Science",
        "cs.LG",
        "lightweight-models",
        "online-learning",
        "optimization-methods",
        "representation-learning",
        "self-supervised-learning",
        "statistical-learning",
        "unsupervised-learning"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 3,
      "github_url": "https://github.com/ggluo/Self-Diffusion",
      "distance": 1
    },
    {
      "id": "019a2902-0d61-7809-a892-6fcd9e0b6bf0",
      "paper_group_id": "019a2902-0d61-7809-a892-6fcd9e0b6bf0",
      "title": "ACG: Action Coherence Guidance for Flow-based VLA models",
      "abstract": "扩散和流匹配模型已成为强大的机器人策略，使视觉-语言-动作（VLA）模型能够在多样的场景和指令中推广。然而，当通过模仿学习进行训练时，其高生成能力使其对人类演示中的噪声敏感：抖动、停顿和抖动降低了动作的一致性。降低的动作一致性导致部署期间的不稳定和轨迹漂移，这在精细操作中是灾难性的，因为精度至关重要。在本文中，我们为VLA模型提出了动作一致性指导（ACG），这是一种无需训练的测试时指导算法，它改善了动作一致性，从而带来了性能提升。在RoboCasa、DexMimicGen和真实世界SO-101任务上进行评估时，ACG始终提高了动作一致性，并提升了各种操作任务的成功率。代码和项目页面可在此https URL和此https URL获得。",
      "paper_summary": null,
      "image_url": "image/2510.22201v1.png",
      "universal_paper_id": "2510.22201",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 18,
          "last_7_days": 18
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-25T07:44:33.000Z",
      "publication_date": "2025-10-25T07:44:33.000Z",
      "updated_at": "2025-10-28T04:09:51.713Z",
      "topics": [
        "Computer Science",
        "cs.RO"
      ],
      "organization_info": [
        {
          "name": "KAIST",
          "image": "images/organizations/kaist.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a2438-eb0a-7ebd-90f3-1c53351e35be",
      "paper_group_id": "019a2438-eb0a-7ebd-90f3-1c53351e35be",
      "title": "Sparser Block-Sparse Attention via Token Permutation",
      "abstract": "扩展大语言模型（LLMs）的上下文长度提供了显著的好处，但计算开销很大。这种开销主要源于自注意力机制，其在序列长度上的复杂度为 $O(N^2)$，这对于内存和延迟都构成了主要瓶颈。幸运的是，注意力矩阵通常是稀疏的，尤其是在较长的序列中，这为优化提供了机会。块稀疏注意力成为了一种有前景的解决方案，它将序列分割成块，并跳过部分块的计算。然而，该方法的有效性高度依赖于底层的注意力模式，这可能导致子最优的块级稀疏性。例如，用于查询的单个块中的重要关键标记可能分散在许多其他块中，从而导致计算冗余。在本研究中，我们提出了排列块稀疏注意力（\\textbf{PBS-Attn}），这是一种即插即用的方法，利用注意力的排列特性来增加块级稀疏性并提高LLM预填充的计算效率。我们在具有挑战性的实际长上下文数据集上进行了全面实验，证明PBS-Attn在模型准确性方面始终优于现有的块稀疏注意力方法，且与完整注意力基线表现相近。凭借我们定制的排列FlashAttention内核，PBS-Attn在长上下文预填充中实现了高达 $2.75\\times$ 的端到端加速，验证了其实际可行性。代码可在此链接获取。",
      "paper_summary": {
        "summary": "Researchers from Fudan University, China Unicom, and ByteDance developed Permuted Block-Sparse Attention (PBS-Attn), an optimization for Large Language Models that rearranges input tokens to increase block-level sparsity. This method achieves competitive accuracy with full attention while providing up to a 2.75x end-to-end speedup for LLM prefilling on long contexts.",
        "originalProblem": [
          "The self-attention mechanism in Large Language Models (LLMs) exhibits quadratic computational and memory complexity, severely bottlenecking performance for long context windows during prefilling.",
          "Existing block-sparse attention methods, designed to reduce this complexity, often suffer from sub-optimal block-level sparsity because critical key tokens for queries are scattered across numerous blocks, forcing more computations than necessary.",
          "The effectiveness of current block selection algorithms is limited by the inherent, often diffuse, attention patterns of raw input sequences, preventing maximal sparsity gains."
        ],
        "solution": [
          "PBS-Attn introduces a token permutation strategy to reorder query, key, and value sequences, clustering important interactions to enhance block-level sparsity before block-sparse attention is applied.",
          "A segmented permutation approach is employed, applying permutations only within predefined segments to rigorously preserve the causal attention mechanism critical for auto-regressive LLMs.",
          "The method uses a query-aware key permutation, where key tokens are sorted within segments based on estimated global importance scores, concentrating attention mass into fewer blocks.",
          "Custom `permuted-FlashAttention` kernels were developed using Triton to efficiently execute both the permutation and subsequent sparse attention computations on GPUs."
        ],
        "keyInsights": [
          "Permuting tokens in the input sequence can inherently increase the block-level sparsity of the attention matrix, thereby making existing block-sparse attention methods more efficient.",
          "Segmented permutation allows for intra-segment token reordering while maintaining inter-segment causality, which is crucial for auto-regressive LLMs, enabling significant sparsity gains without violating model integrity.",
          "Query-aware key permutation, particularly for grouped-query attention (GQA) models, effectively clusters important key-value pairs, leading to a better performance-density trade-off compared to permuting queries or both."
        ],
        "results": [
          "PBS-Attn achieved the best overall accuracy among all block-sparse attention methods on LongBench and LongBenchv2, closely matching the performance of the full attention baseline (e.g., Llama-3.1-8B score of 37.37 vs. Full Attention's 38.28).",
          "The method delivered an impressive 2.75x end-to-end speedup for LLM prefilling at 256K context length, and consistent speedups across various context lengths (8K to 512K), outperforming competing sparse attention techniques.",
          "Permutation led to a tangible increase in block-level sparsity (e.g., a 7% absolute improvement at 8K context length) with minimal overhead, accounting for only 1.3% of the total FlashAttention time at 128K context."
        ]
      },
      "image_url": "image/2510.21270v1.png",
      "universal_paper_id": "2510.21270",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 53,
          "last_7_days": 53
        },
        "public_total_votes": 8
      },
      "first_publication_date": "2025-10-24T09:11:50.000Z",
      "publication_date": "2025-10-24T09:11:50.000Z",
      "updated_at": "2025-10-27T05:51:41.322Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "efficient-transformers",
        "inference-optimization",
        "lightweight-models",
        "sequence-modeling",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 4,
      "github_url": "https://github.com/xinghaow99/pbs-attn",
      "distance": 1
    },
    {
      "id": "019a28e7-aac4-7f52-8cae-ca97a6eec16d",
      "paper_group_id": "019a28e7-aac4-7f52-8cae-ca97a6eec16d",
      "title": "LimRank: Less is More for Reasoning-Intensive Information Reranking",
      "abstract": "现有的方法通常依赖大规模的精细调优来适应大语言模型（LLM）用于信息重排名任务，这在计算上是非常昂贵的。在这项工作中，我们证明了现代LLM可以仅通过极少量的高质量监督有效地进行适应。为此，我们设计了LIMRANK-SYNTHESIZER，一个可重用的开源管道，用于生成多样、具有挑战性且现实的重排名示例。使用这些合成数据，我们对我们的重排名模型LIMRANK进行了精细调优。我们在两个具有挑战性的基准上评估了LIMRANK，即用于推理密集型检索的BRIGHT和用于遵循指令的FollowIR。我们的实验表明，LIMRANK的性能具有竞争力，同时其训练数据量不到以往研究中使用数据的5%。进一步的消融研究表明，LIMRANK-SYNTHESIZER的有效性以及LIMRANK在下游任务中的强泛化能力，包括科学文献检索和用于知识密集型问题解决的检索增强生成。",
      "paper_summary": null,
      "image_url": "image/2510.23544v1.png",
      "universal_paper_id": "2510.23544",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 17,
          "last_7_days": 17
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-27T17:19:37.000Z",
      "publication_date": "2025-10-27T17:19:37.000Z",
      "updated_at": "2025-10-28T03:41:02.532Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "cs.IR",
        "data-curation",
        "fine-tuning",
        "instruction-tuning",
        "reasoning",
        "synthetic-data",
        "text-classification",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Yale University",
          "image": "images/organizations/yale.png"
        }
      ],
      "author_info": [],
      "github_stars": 1,
      "github_url": "https://github.com/SighingSnow/limrank",
      "distance": 1
    },
    {
      "id": "019a2564-b02d-723a-be7f-36b6fde61832",
      "paper_group_id": "019a2564-b02d-723a-be7f-36b6fde61832",
      "title": "AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite",
      "abstract": "AI代理有潜力通过自动化文献综述、复现实验、分析数据，甚至提出新的研究方向，来彻底改变科学生产力；事实上，现在已经有许多这样的代理，从通用的“深度研究”系统到专门的科学特定代理，例如AI科学家和AIGS。对这些代理进行严格评估对于进展至关重要。然而，现有的基准测试在多个方面存在不足：（1）未能提供全面的、以产品为导向的实际应用案例的衡量，例如科学研究；（2）缺乏可重复的代理工具，无法进行核心代理能力的受控比较；（3）未考虑混杂变量，如模型成本和工具访问；（4）未提供标准化接口以便快速进行代理原型设计和评估；（5）缺乏全面的基线代理以识别真正的进展。为此，我们定义了更严格地基准测试代理的原则和工具。基于这些原则，我们推出了AstaBench，这是一个提供代理能力在科学研究中执行能力的首次全面衡量的工具包，包含2400多个问题，涵盖整个科学发现过程和多个科学领域，并包括许多受实际用户请求启发的问题。我们的工具包配备了首个具有生产级搜索工具的科学研究环境，使得评估更具控制性和可重复性，从而更好地考虑混杂因素。此外，我们还提供了九类为科学优化的Asta代理的全面套件和众多基准。在对57个代理进行评估的过程中，涉及22个代理类别，我们发现了一些有趣的结果，最重要的是，尽管在某些个别方面取得了显著进展，但人工智能距离解决科学研究助手的挑战仍然相距甚远。",
      "paper_summary": null,
      "image_url": "image/2510.21652v1.png",
      "universal_paper_id": "2510.21652",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 22,
          "last_7_days": 22
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-24T17:10:26.000Z",
      "publication_date": "2025-10-24T17:10:26.000Z",
      "updated_at": "2025-10-27T11:19:07.053Z",
      "topics": [
        "agentic-frameworks",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "data-curation",
        "ml-systems",
        "reasoning",
        "test-time-inference",
        "tool-use"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": 40,
      "github_url": "https://github.com/allenai/asta-bench",
      "distance": 1
    }
  ],
  "page": 0
};