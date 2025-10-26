const papersData = {
  "papers": [
    {
      "id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "paper_group_id": "019a09db-6f76-753a-a087-dfaa1ae3c0d3",
      "title": "DeepSeek-OCR: Contexts Optical Compression",
      "abstract": "我们呈现了DeepSeek-OCR，这是对通过光学二维映射压缩长文本的可行性进行的初步研究。DeepSeek-OCR由两个组件组成：DeepEncoder和作为解码器的DeepSeek3B-MoE-A570M。具体而言，DeepEncoder作为核心引擎，旨在在高分辨率输入下保持低激活，同时实现高压缩比，以确保最佳和可管理的视觉标记数量。实验表明，当文本标记数量在视觉标记的10倍以内（即压缩比<10x）时，该模型能够实现97%的解码（OCR）精度。即使在压缩比达到20x的情况下，OCR准确率仍保持在约60%。这为历史长文本压缩以及大语言模型中的记忆遗忘机制等研究领域带来了相当大的潜力。此外，DeepSeek-OCR还展示了很高的实践价值。在OmniDocBench上，它仅使用100个视觉标记就超越了GOT-OCR2.0（256个标记/页），并在使用不到800个视觉标记的情况下超过了MinerU2.0（每页平均6000多个标记）。在生产中，DeepSeek-OCR可以以每天超过20万页的规模为大语言模型/视觉语言模型生成训练数据（单个A100-40G）。代码和模型权重可在此HTTP网址公开获取。",
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
        "total_votes": 188,
        "visits_count": {
          "all": 5388,
          "last_7_days": 5388
        },
        "public_total_votes": 331
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
      "id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "paper_group_id": "019a16fc-ffec-7205-bcf2-e9fe4b378109",
      "title": "Real Deep Research for AI, Robotics and Beyond",
      "abstract": "随着人工智能和机器人研究的快速增长，年产超过10,000篇论文，研究人员保持最新信息变得越来越困难。快速发展的趋势、跨学科工作的兴起以及探索超出自身专长领域的必要性都加剧了这一挑战。为了应对这些问题，我们提出了一种通用的管道，能够系统地分析任何研究领域：识别新兴趋势、发现跨领域机会，并为新的研究提供具体的切入点。在这项工作中，我们呈现了真实深度研究（Real Deep Research，RDR）这一综合框架，适用于人工智能和机器人领域，特别关注基础模型和机器人技术的进展。我们还简要扩展了对其他科学领域的分析。主要论文详细介绍了RDR管道的构建，而附录则提供了针对每个分析主题的广泛结果。我们希望这项工作能为在人工智能及其他领域工作的研究人员提供启示。",
      "paper_summary": null,
      "image_url": "image/2510.20809v1.png",
      "universal_paper_id": "2510.20809",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 181,
          "last_7_days": 181
        },
        "public_total_votes": 21
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
      "id": "019a04ba-21a1-7422-b529-6edb25aa0e24",
      "paper_group_id": "019a04ba-21a1-7422-b529-6edb25aa0e24",
      "title": "The Free Transformer",
      "abstract": "我们提出了一种解码器Transformer的扩展，将其生成过程基于随机潜变量，这些潜变量通过变分过程在无监督的情况下学习。实验评估表明，允许这种条件设置在下游任务上带来了显著的提升。",
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
        "total_votes": 33,
        "visits_count": {
          "all": 1590,
          "last_7_days": 1590
        },
        "public_total_votes": 116
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
      "id": "019a0eb1-dded-7f4c-a639-99edbbeee49e",
      "paper_group_id": "019a0eb1-dded-7f4c-a639-99edbbeee49e",
      "title": "Loopholing Discrete Diffusion: Deterministic Bypass of the Sampling Wall",
      "abstract": "离散扩散模型通过并行解码提供了一种有前景的自回归生成替代方案，但它们面临采样壁垒：一旦发生类别采样，丰富的分布信息就会崩溃成独热向量，无法在步骤之间传播，从而迫使后续步骤在有限的信息下操作。为了解决这个问题，我们引入了一种名为Loopholing的新颖简单机制，通过确定性的潜在路径来保留这些信息，从而形成了Loopholing离散扩散模型（LDDMs）。使用自条件策略高效训练的LDDMs在生成困惑度上取得了显著提升，相较于之前的基线降低了多达61%，缩小了（在某些情况下超越了）与自回归模型的差距，并生成了更连贯的文本。在推理任务中，LDDMs在像Countdown和Game of 24这样的算术基准测试上也改善了表现。这些结果还表明，loopholing减少了闲置步骤和振荡，为高质量的非自回归文本生成提供了一条可扩展的路径。",
      "paper_summary": {
        "summary": "This research introduces Loopholing, a mechanism that deterministically propagates rich continuous latent information across denoising steps in discrete diffusion models, directly addressing the \"sampling wall\" problem. Loopholing Discrete Diffusion Models (LDDMs) demonstrate enhanced language generation quality and improved performance on reasoning tasks, often surpassing autoregressive baselines in generative perplexity and consistency.",
        "originalProblem": [
          "Discrete diffusion models suffer from the \"sampling wall problem,\" where rich distributional information is discarded after sampling a one-hot token, leading to inefficient denoising.",
          "This information loss results in \"idle steps\" (no progress) and \"temporal oscillation\" (tokens flipping back and forth) during the iterative generation process.",
          "Empirically, discrete diffusion models have lagged behind autoregressive models in generation quality despite their theoretical advantages in parallel generation."
        ],
        "solution": [
          "Loopholing introduces a deterministic latent pathway, directly passing a continuous latent vector (rich contextual state) to the subsequent denoising step, alongside the standard stochastic one-hot token.",
          "The input to each denoising step fuses the current one-hot token embedding with the previous step's propagated continuous latent state via Layer Normalization.",
          "A self-conditioning strategy is employed during training to handle the recurrent dependency of propagated latents efficiently, using a two-pass approach with a stop-gradient operator to prevent costly temporal unrolling."
        ],
        "keyInsights": [
          "Preserving and propagating rich continuous latent information across denoising steps is crucial for mitigating the sampling wall problem and improving the stability and quality of discrete diffusion models.",
          "The deterministic latent pathway in Loopholing effectively provides a recurrent memory, akin to hidden states in RNNs, enabling the model to make more informed and consistent denoising decisions.",
          "Self-conditioning allows for efficient training of models with recurrent latent dependencies without the computational burden of full backpropagation through time."
        ],
        "results": [
          "LDDM-M reduced MDLM's OWT validation perplexity from 23.05 to 21.90 and achieved a 55% reduction in Generative Perplexity (49.13 vs 108.94 at 1024 steps).",
          "LDDM-U showed a 61% reduction in Generative Perplexity (28.76 vs 73.95) over UDLM, surpassing the autoregressive baseline after 512 steps.",
          "On reasoning tasks, the 85M parameter LDDM-G improved accuracy on Game of 24 by 16% and Countdown 4 by almost 8% over the MGDM baseline, attributed to better preservation of contextual ambiguity."
        ]
      },
      "image_url": "image/2510.19304v1.png",
      "universal_paper_id": "2510.19304",
      "metrics": {
        "total_votes": 16,
        "visits_count": {
          "all": 486,
          "last_7_days": 486
        },
        "public_total_votes": 55
      },
      "first_publication_date": "2025-10-22T07:08:47.000Z",
      "publication_date": "2025-10-22T07:08:47.000Z",
      "updated_at": "2025-10-23T01:32:09.069Z",
      "topics": [
        "Computer Science",
        "cs.LG"
      ],
      "organization_info": [
        {
          "name": "KAIST",
          "image": "images/organizations/kaist.png"
        },
        {
          "name": "New York University",
          "image": "images/organizations/nyu.png"
        },
        {
          "name": "Microsoft",
          "image": "images/organizations/microsoft.png"
        },
        {
          "name": "EPFL",
          "image": "images/organizations/epfl.png"
        },
        {
          "name": "SAP",
          "image": null
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
      "abstract": "尽管大型语言模型（LLMs）具有卓越的能力，但在动态复杂的环境中，它们很难有效利用历史交互信息。记忆系统使LLMs能够超越无状态交互，通过引入持久信息存储、检索和利用机制。然而，现有的记忆系统往往会引入相当大的时间和计算开销。为此，我们提出了一种新型记忆系统，称为LightMem，它在记忆系统的性能和效率之间取得了平衡。LightMem受到阿特金森-希夫林人类记忆模型的启发，将记忆组织为三个互补阶段。首先，受到认知启发的感官记忆通过轻量级压缩快速过滤不相关的信息，并根据主题对信息进行分组。接下来，主题意识的短期记忆巩固这些基于主题的组，组织和总结内容以便更结构化的访问。最后，带有睡眠时间更新的长期记忆采用离线过程，将巩固与在线推理解耦。基于GPT和Qwen的LongMemEval实验表明，LightMem在准确性上优于强基线（最高提高10.9%），同时将令牌使用减少了最多117倍，API调用减少了最多159倍，运行时间减少了超过12倍。代码可在此https URL获取。",
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
        "total_votes": 8,
        "visits_count": {
          "all": 575,
          "last_7_days": 575
        },
        "public_total_votes": 59
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
      "id": "019a09ac-49fc-7609-97c6-c538beab772c",
      "paper_group_id": "019a09ac-49fc-7609-97c6-c538beab772c",
      "title": "How Do LLMs Use Their Depth?",
      "abstract": "越来越多的证据表明，大型语言模型在使用其层次深度时并不均匀，但我们仍然缺乏对其分层预测动态的细致理解。在本论文中，我们追踪了多个开放权重模型在推理过程中的中间表示，揭示了深度的结构性和细微使用。具体而言，我们提出了一个“先猜测后精炼”的框架，解释了大型语言模型（LLMs）如何在内部结构化其计算以进行预测。我们首先显示，在早期的LLM层中，排名靠前的预测主要由高频标记组成，这些标记由于缺乏适当的上下文信息，而被模型早期提出作为统计猜测。随着上下文信息的逐渐深入，这些初步猜测被精炼为上下文合适的标记。即使是早期层的高频标记预测也有超过70%的时间得到精炼，这表明正确的标记预测并不是“一成不变”的。然后，我们超越基于频率的预测，考察在三项案例研究中层深度的动态使用。(i) 词性分析表明，功能词在平均上是最早被正确预测的。(ii) 事实回忆任务分析表明，在一个多标记答案中，第一个标记需要的计算深度超出了其余部分。(iii) 多项选择任务分析显示，模型在前一半的层中识别答复的格式，但最终确定其答复则是在接近末尾的时候。综上所述，我们的结果提供了LLMs深度使用的详细视角，揭示了成功预测背后的逐层计算，并为未来的研究提供了改善基于变换器模型的计算效率的见解。",
      "paper_summary": {
        "summary": "Researchers quantified how large language models utilize their architectural depth, revealing a 'guess-then-refine' process where early layers propose statistical guesses and deeper layers contextually refine them. They also found that LLMs adaptively allocate computational depth based on prediction complexity, resolving easier tokens earlier than complex ones.",
        "originalProblem": [
          "A lack of fine-grained, quantitative understanding exists regarding how Large Language Models (LLMs) specifically leverage their architectural depth layer-by-layer during inference.",
          "It is unclear whether LLMs make specific token predictions solely at the final layer or if intermediate layers play a role in developing and refining these predictions.",
          "The extent to which LLMs use their computational depth uniformly across all tasks and tokens, versus adaptively based on complexity, was previously unknown."
        ],
        "solution": [
          "The study leverages the TunedLens probe to faithfully decode intermediate layer representations into the token space, ensuring robust layer-wise interpretation.",
          "It proposes and empirically validates a 'Guess-then-Refine' framework, characterizing LLM inference as an iterative process of initial statistical guesses followed by contextual refinement.",
          "The work investigates 'Complexity-Aware Depth Use' through detailed case studies on next-token prediction, multi-token factual recall, and constrained-choice downstream tasks across various LLM architectures."
        ],
        "keyInsights": [
          "LLMs inherently operate using a 'guess-then-refine' mechanism, where early layers tend to propose high-frequency, statistically optimal guesses, which are then substantially refined by deeper layers as more context is integrated.",
          "LLMs exhibit 'complexity-aware depth use,' meaning they dynamically allocate computational resources: shallower depths for 'easier' subtasks like predicting function words or identifying valid options, and deeper layers for 'harder' subtasks like predicting content words or initiating multi-token factual responses.",
          "Current early-exiting strategies might interfere with the LLM's natural refinement dynamics, suggesting that more sophisticated, adaptive exit mechanisms are needed to balance efficiency and prediction accuracy effectively."
        ],
        "results": [
          "Early layers (e.g., layer 1 for Pythia-6.9B) show over 75% of top-ranked predictions belonging to high-frequency tokens, with about 80% of these initial guesses from the top frequency bucket being overturned by the final layer.",
          "Function words (e.g., determiners, adpositions) reach rank-1 predictions significantly earlier (average around layer 5) than content words (e.g., adjectives, verbs, nouns) which require deeper processing (average around layer 20).",
          "For multi-token factual answers, the first token requires substantially more computational depth (e.g., Pythia-6.9B, layer 27) than subsequent tokens (e.g., layers 20 and 12 for the second and third tokens, respectively)."
        ]
      },
      "image_url": "image/2510.18871v1.png",
      "universal_paper_id": "2510.18871",
      "metrics": {
        "total_votes": 7,
        "visits_count": {
          "all": 270,
          "last_7_days": 270
        },
        "public_total_votes": 33
      },
      "first_publication_date": "2025-10-21T17:59:05.000Z",
      "publication_date": "2025-10-21T17:59:05.000Z",
      "updated_at": "2025-10-22T02:07:57.436Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "efficient-transformers",
        "mechanistic-interpretability",
        "reasoning",
        "test-time-inference",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "UC Berkeley",
          "image": "images/organizations/berkeley.png"
        },
        {
          "name": "Georgia Institute of Technology",
          "image": "images/organizations/georgiatech.png"
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
      "abstract": "使用世界模型进行规划为机器人控制提供了一种强大的范式。传统的方法训练一个模型，根据当前帧和动作预测未来帧，然后可以用来进行规划。然而，预测未来像素的目标通常与实际的规划目标相悖；强大的像素重建并不总是与良好的规划决策相关联。本文提出，与其重建未来帧的像素，不如让世界模型仅预测与任务相关的未来语义信息。为此，本文将世界建模视为关于未来帧语义信息的视觉问答问题。这种视角使得可以使用与视觉语言模型相同的工具来处理世界建模。因此，通过在图像-动作-文本数据上的监督微调过程，视觉语言模型可以被训练为“语义”世界模型，从而在决策制定中实现规划，同时继承许多预训练视觉语言模型的泛化和鲁棒性特性。论文展示了如何使用这样的语义世界模型在开放式机器人任务上进行策略改进，相较于典型的基于重建的动作条件世界建模方法，显著提高了泛化能力。网站链接可在此https URL找到。",
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
        "total_votes": 5,
        "visits_count": {
          "all": 129,
          "last_7_days": 129
        },
        "public_total_votes": 23
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
      "id": "019a13f2-71bc-722c-932a-fc8ebf018ff5",
      "paper_group_id": "019a13f2-71bc-722c-932a-fc8ebf018ff5",
      "title": "LLM-empowered knowledge graph construction: A survey",
      "abstract": "知识图谱（KGs）长期以来一直作为结构化知识表示和推理的基本基础设施。随着大型语言模型（LLMs）的出现，知识图谱的构建进入了一个新的范式——从基于规则和统计的流程转变为基于语言和生成框架。本综述提供了关于LLM驱动的知识图谱构建的最新进展的全面概述，系统分析了LLM如何重塑经典的三层管道，即本体工程、知识提取和知识融合。\n\n我们首先回顾传统的知识图谱方法，以建立概念基础，然后从两个互补的角度回顾新兴的LLM驱动方法：基于模式的范式，强调结构、标准化和一致性；以及无模式范式，突出灵活性、适应性和开放发现。在每个阶段，我们综合了具有代表性的框架，分析了它们的技术机制，并识别其局限性。\n\n最后，本综述概述了关键趋势和未来研究方向，包括基于KG的LLM推理、代理系统的动态知识记忆以及多模态知识图谱构建。通过这一系统性回顾，我们旨在阐明LLM与知识图谱之间不断发展的相互作用，桥接符号知识工程与神经语义理解，为开发自适应、可解释和智能的知识系统铺平道路。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 57,
          "last_7_days": 57
        },
        "public_total_votes": 11
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
      "id": "019a13e9-d98d-7ae9-855d-24e7a7abad50",
      "paper_group_id": "019a13e9-d98d-7ae9-855d-24e7a7abad50",
      "title": "Collective Communication for 100k+ GPUs",
      "abstract": "大型语言模型（LLMs）规模的不断扩大，需要高效的集体通信框架，尤其是当训练工作负载扩展到数十万个GPU时。传统通信方法在这一规模下面临显著的吞吐量和延迟限制，妨碍了最先进模型的开发和部署。本文提出了在Meta开发的NCCLX集体通信框架，旨在优化整个LLM生命周期的性能，从大规模训练的同步需求到推理的低延迟要求。该框架设计支持超过100,000个GPU的集群上的复杂工作负载，确保可靠的高吞吐量和低延迟的数据交换。对Llama4模型的实证评估表明通信效率有显著提高。本研究为使下一代LLMs在前所未有的规模下运行提供了一个强健的解决方案。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 60,
          "last_7_days": 60
        },
        "public_total_votes": 13
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
      "id": "019a0ec5-76e2-7f82-ab67-f687c4ef5018",
      "paper_group_id": "019a0ec5-76e2-7f82-ab67-f687c4ef5018",
      "title": "Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing",
      "abstract": "最近，多模态模型的进展展示了显著的文本引导图像编辑能力，像GPT-4o和Nano-Banana这样的系统设定了新的基准。然而，研究界的进展仍然受到缺乏大规模、高质量和公开可访问的真实图像数据集的限制。我们引入了Pico-Banana-400K，这是一个全面的40万图像数据集，旨在基于指令进行图像编辑。我们的数据集是通过利用Nano-Banana从OpenImages集合中的真实照片生成多样的编辑对而构建的。Pico-Banana-400K与之前的合成数据集的区别在于我们对质量和多样性采取了系统化的方法。我们采用了细粒度的图像编辑分类法，以确保编辑类型的全面覆盖，同时通过基于MLLM的质量评分和精心策划来保持内容的准确保留和指令的忠实性。除了单回合编辑，Pico-Banana-400K还支持对复杂编辑场景的研究。该数据集包含三个专门子集：(1) 一个72K示例的多回合集合，用于研究连续修改中的顺序编辑、推理和规划；(2) 一个56K示例的偏好子集，用于对齐研究和奖励模型训练；(3) 配对的长短编辑指令，用于发展指令重写和摘要能力。通过提供这一大规模、高质量和任务丰富的资源，Pico-Banana-400K为训练和基准测试下一代文本引导图像编辑模型奠定了坚实的基础。",
      "paper_summary": {
        "summary": "Apple researchers introduced Pico-Banana-400K, a large-scale dataset of approximately 400,000 text-guided image editing examples built from real OpenImages photographs. This resource features MLLM-based quality assessment and includes specialized subsets for multi-turn editing and preference learning to advance model training for improved instruction following and visual fidelity.",
        "originalProblem": [
          "There is a lack of large-scale, high-quality, and openly accessible datasets derived from real images for instruction-based image editing.",
          "Existing datasets often suffer from synthetic origins, limited scale, or heavy reliance on expensive human curation, leading to domain shifts and inconsistent quality.",
          "Data for advanced editing scenarios, such as multi-turn interactions and explicit preference learning for human alignment, has been scarce."
        ],
        "solution": [
          "Constructed Pico-Banana-400K using real photographs from the OpenImages dataset, covering 35 distinct edit types organized under a comprehensive taxonomy.",
          "Leveraged state-of-the-art MLLMs (Gemini-2.5-Flash and Qwen2.5-7B-Instruct) to generate both detailed and concise natural language instructions for each edit.",
          "Implemented an automated pipeline with Nano-Banana (Gemini-2.5-Flash-Image) for image editing and Gemini-2.5-Pro as an MLLM-based judge for rigorous quality assessment and a retry mechanism, generating preference data from successful and failed edits."
        ],
        "keyInsights": [
          "Automated MLLM-based pipelines can effectively scale the generation and quality control of large-scale, high-fidelity datasets for complex multimodal tasks like text-guided image editing.",
          "The inclusion of specialized subsets for multi-turn editing and preference learning is crucial for developing models capable of handling iterative, context-aware, and human-aligned editing interactions.",
          "Current state-of-the-art image editing models demonstrate varying performance across different edit types, with precise geometric manipulations, object relocation, and text editing posing the greatest challenges."
        ],
        "results": [
          "Successfully compiled Pico-Banana-400K, a dataset comprising approximately 400,000 high-quality text-guided image editing examples from real-world imagery.",
          "The dataset is structured into a 258K single-turn supervised fine-tuning subset, a 56K preference subset for alignment research, and a 72K multi-turn editing subset for sequential reasoning.",
          "Analysis of per-edit type success rates revealed that global appearance and stylistic transformations achieve high reliability, while operations requiring fine spatial control, such as object relocation and text manipulation, exhibit the lowest success rates (e.g., \"Relocate an object\" at 0.5923 and \"Change font style or color of visible text\" at 0.5759)."
        ]
      },
      "image_url": "image/2510.19808v1.png",
      "universal_paper_id": "2510.19808",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 199,
          "last_7_days": 199
        },
        "public_total_votes": 28
      },
      "first_publication_date": "2025-10-22T17:43:15.000Z",
      "publication_date": "2025-10-22T17:43:15.000Z",
      "updated_at": "2025-10-23T01:53:33.410Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "data-curation",
        "fine-tuning",
        "generative-models",
        "image-generation",
        "instruction-tuning",
        "multi-modal-learning",
        "reasoning",
        "reinforcement-learning",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "Apple",
          "image": "images/organizations/apple.png"
        }
      ],
      "author_info": [],
      "github_stars": 162,
      "github_url": "https://github.com/apple/pico-banana-400k",
      "distance": 1
    },
    {
      "id": "019a1759-3897-7864-9e1c-c9fbe9ad8173",
      "paper_group_id": "019a1759-3897-7864-9e1c-c9fbe9ad8173",
      "title": "Thought Communication in Multiagent Collaboration",
      "abstract": "自然语言长期以来促进了人类合作，但其损失性、模糊性和间接性限制了集体智慧的潜力。虽然机器不受这些限制，但大多数基于大型语言模型的多智能体系统仍仅依赖自然语言，交换令牌或其嵌入。为了超越语言，我们引入了一种新范式，称为思想交流，这种方式使智能体能够直接进行心灵对心灵的互动，类似于超感知。为了以一种有原则的方式揭示这些潜在思想，我们将这一过程形式化为一个一般的潜变量模型，其中智能体状态是由潜在思想的未知函数生成的。我们证明，在没有辅助信息的非参数设置下，任何一对智能体之间的共享和私人潜在思想都可以被识别。此外，思想共享的全局结构，包括哪些智能体共享哪些思想以及这些关系如何构建，也可以在理论上得到恢复。在建立的理论指导下，我们开发了一个框架，从所有智能体提取潜在思想，并在交流之前分配给每个智能体相关思想及其共享模式。这个范式自然地扩展到所有模态，因为大多数观测数据来自隐藏的生成过程。对合成和真实世界基准的实验验证了理论，并展示了思想交流的协作优势。我们希望这项工作能够照亮利用隐藏世界的潜力，因为许多挑战仍然无法仅通过表面观察来解决，无论计算或数据规模如何。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 66,
          "last_7_days": 66
        },
        "public_total_votes": 10
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
      "id": "019a13ec-c73a-71df-aa73-6eb0795f4098",
      "paper_group_id": "019a13ec-c73a-71df-aa73-6eb0795f4098",
      "title": "KL-Regularized Reinforcement Learning is Designed to Mode Collapse",
      "abstract": "人们普遍认为，优化反向KL散度会导致“模式寻求”，而优化正向KL散度则会导致“质量覆盖”，如果目标是从多个不同模式中采样，后者通常更受青睐。我们通过数学和实证研究表明，这种直觉并不一定适用于带有反向/正向KL正则化的强化学习（例如，常用于语言模型中）。相反，反向/正向KL的选择决定了最优目标分布的家族，这些分布由正则化系数进行参数化。模式覆盖主要依赖于其他因素，例如正则化强度以及奖励和参考概率之间的相对比例。此外，我们表明，常用的设置如低正则化强度和相等的可验证奖励往往指定单模态目标分布，这意味着优化目标在构建上是非多样的。我们利用这些洞见构建了一个简单、可扩展且理论上有依据的算法。它对奖励大小的改变很小，但优化了一个将高概率分配给所有高质量采样模式的目标分布。在实验中，这一简单的修改能够使大型语言模型和化学语言模型进行后训练，以提高解决方案的质量和多样性，而无需任何外部多样性信号，并且在使用反向和正向KL时都能发挥作用，而纯粹的尝试则失败。",
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
        "total_votes": 1,
        "visits_count": {
          "all": 44,
          "last_7_days": 44
        },
        "public_total_votes": 11
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
      "id": "019a1962-2af6-7dc8-abba-d7cc981b596b",
      "paper_group_id": "019a1962-2af6-7dc8-abba-d7cc981b596b",
      "title": "Positional Encoding Field",
      "abstract": "扩散变换器（DiTs）已成为视觉生成的主流架构，为最先进的图像和视频模型提供支持。通过将图像表示为带有位置编码（PEs）的补丁令牌，DiTs结合了变换器的可扩展性与空间和时间的归纳偏置。在本研究中，我们重新审视了DiTs如何组织视觉内容，并发现补丁令牌表现出惊人的独立性：即使当PEs受到扰动时，DiTs仍然能够生成全局一致的输出，这表明空间一致性主要由PEs控制。受这一发现的启发，我们引入了位置编码场（PE-Field），将位置编码从二维平面扩展到结构化的三维场。PE-Field结合了深度感知编码以进行体积推理，以及分层编码以实现细粒度的子补丁控制，使DiTs能够直接在三维空间中建模几何形状。我们的PE-Field增强的DiT在单图像新视图合成上实现了最先进的性能，并可推广到可控的空间图像编辑。",
      "paper_summary": {
        "summary": "Researchers from the University of Texas at Austin and Pixocial Technology developed the Positional Encoding Field (PE-Field), an extension for Diffusion Transformers (DiTs) that imbues them with geometry-aware generative capabilities for visual tasks. This approach achieved state-of-the-art performance in single-image novel view synthesis and demonstrated versatility in controllable spatial image editing by leveraging depth-aware and multi-level positional encodings.",
        "originalProblem": [
          "Diffusion Transformers (DiTs) using standard 2D positional encodings lack intrinsic 3D reasoning capabilities, limiting their application in tasks like Novel View Synthesis (NVS).",
          "Existing single-image NVS methods often struggle with precise camera pose control, geometric consistency, and introduce artifacts, or rely on computationally intensive intermediate 3D representations or video generation.",
          "Current DiT-based image editing models, while flexible, provide limited fine-grained, geometry-aware spatial control, frequently altering content or offering only minor viewpoint adjustments."
        ],
        "solution": [
          "A Positional Encoding Field (PE-Field) was introduced to extend traditional 2D positional encodings, incorporating 3D depth information into the DiT architecture.",
          "Multi-level positional encodings were integrated into the Multi-Head Self-Attention (MHA) blocks of the DiT, allowing different heads to process spatial information at varying granularities for sub-patch detail modeling.",
          "A depth-aware Rotary Positional Encoding (RoPE) was designed, partitioning embedding vectors to encode x, y, and z coordinates independently, enabling volumetric reasoning and geometric consistency."
        ],
        "keyInsights": [
          "Patch tokens within Diffusion Transformers exhibit a surprising degree of independence, with global spatial coherence predominantly governed by positional encodings rather than complex token interactions.",
          "Manipulating positional encodings alone can effectively induce structured reorganization of spatial content in DiTs, opening a new paradigm for spatially controllable generation.",
          "Augmenting DiTs with explicit 3D and hierarchical positional information allows them to natively reason about geometry, transforming them into geometry-aware generative architectures without explicit 3D representations."
        ],
        "results": [
          "The PE-Field augmented DiT achieved state-of-the-art results in single-image novel view synthesis on Tanks-and-Temples (PSNR 22.12, SSIM 0.732, LPIPS 0.174), RE10K, and DL3DV datasets, outperforming previous methods significantly.",
          "Demonstrated superior controllable spatial image editing, providing more precise viewpoint changes and maintaining content consistency compared to prompt-based models like Flux.1 Kontext and Qwen-Image-Edit.",
          "Ablation studies confirmed that both multi-level and depth-aware positional encodings are critical components, with their removal leading to noticeable image quality degradation and severe spatial misalignment, respectively."
        ]
      },
      "image_url": "image/2510.20385v1.png",
      "universal_paper_id": "2510.20385",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 44,
          "last_7_days": 44
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-23T09:32:37.000Z",
      "publication_date": "2025-10-23T09:32:37.000Z",
      "updated_at": "2025-10-25T03:20:55.286Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "generative-models",
        "geometric-deep-learning",
        "image-generation",
        "neural-rendering",
        "representation-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "University of Texas at Austin",
          "image": "images/organizations/university-of-texas-at-austin.jpeg"
        },
        {
          "name": "Pixocial Technology",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 0,
      "github_url": "https://github.com/christopherwun/NeRF-positional-encoding",
      "distance": 1
    },
    {
      "id": "019a0eca-57da-78b4-9363-48414a186c62",
      "paper_group_id": "019a0eca-57da-78b4-9363-48414a186c62",
      "title": "Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning",
      "abstract": "在本技术报告中，我们介绍了环线性模型系列，特别包括环迷你线性-2.0和环闪电线性-2.0。环迷你线性-2.0包含16亿个参数和9.57亿个激活，而环闪电线性-2.0则包含1040亿个参数和61亿个激活。这两个模型采用了一种混合架构，有效地将线性注意力和softmax注意力结合起来，显著减少了在长上下文推理场景中的输入/输出和计算开销。与320亿参数的稠密模型相比，该系列将推理成本降低到原来的1/10，且与原始环系列相比，成本也降低了超过50%。此外，通过系统性探索混合架构中不同注意力机制的比例，我们已经确定了当前的最优模型结构。此外，通过利用我们自行开发的高性能FP8操作符库linghe，总体训练效率提高了50%。受益于训练和推理引擎操作符之间的高度一致性，这些模型在强化学习阶段能够进行长期、稳定且高效的优化，在多个具有挑战性的复杂推理基准测试中始终保持最先进的性能。",
      "paper_summary": {
        "summary": "The \"Ling Team\" at inclusionAI developed the Ring-linear model series, an efficient hybrid attention architecture that integrates linear and softmax attention with extensive system-level optimizations for long-context reasoning in LLMs. These models achieve significant reductions in inference and training costs while maintaining state-of-the-art performance across complex reasoning benchmarks.",
        "originalProblem": [
          "High computational and I/O resource consumption in LLMs for long contexts due to the quadratic complexity of traditional attention mechanisms.",
          "Pure linear attention models often underperform in complex retrieval tasks and practical industrial scenarios despite their theoretical efficiency.",
          "Challenges in achieving stable and efficient reinforcement learning (RL) alignment due to 'training-inference disparity' and resource demands for post-training."
        ],
        "solution": [
          "Introduced the Ring-linear model series, a highly sparse Mixture-of-Experts (MoE) architecture featuring a Hybrid Linear Attention mechanism with optimized ratios of linear to softmax attention.",
          "Implemented extensive GPU kernel optimizations through a self-developed high-performance FP8 operator library (`linghe`) and an offline inference framework (`Flood`).",
          "Developed a robust two-stage pre-training strategy and a post-training RL alignment method that systematically addresses 'training-inference disparity' across critical LLM modules."
        ],
        "keyInsights": [
          "Hybrid attention architectures, combining linear and softmax attention with specific optimal ratios, provide a superior balance of efficiency and expressive power for long-context LLMs.",
          "Deep computational optimizations at the GPU kernel level and advanced FP8 mixed-precision training are crucial for realizing substantial efficiency gains in both training and inference.",
          "Systematic resolution of 'training-inference disparity' across core LLM modules (e.g., KV Cache, LM Head) is essential for stable, efficient, and high-performing reinforcement learning alignment."
        ],
        "results": [
          "Achieved significant cost reductions, with inference cost reduced to 1/10 compared to a 32B dense model and overall training efficiency improved by 50%.",
          "Demonstrated substantial throughput gains for long-context inference, outperforming Ring-2.0 and dense baselines by factors of 2.5x to 10x for context lengths beyond 8K.",
          "Maintained state-of-the-art (SOTA) performance across challenging complex reasoning benchmarks (e.g., AIME'25, LiveCodeBench) against similarly sized or larger competitor models."
        ]
      },
      "image_url": "image/2510.19338v1.png",
      "universal_paper_id": "2510.19338",
      "metrics": {
        "total_votes": 5,
        "visits_count": {
          "all": 286,
          "last_7_days": 286
        },
        "public_total_votes": 37
      },
      "first_publication_date": "2025-10-22T07:59:38.000Z",
      "publication_date": "2025-10-22T07:59:38.000Z",
      "updated_at": "2025-10-23T01:58:53.146Z",
      "topics": [
        "attention-mechanisms",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "efficient-transformers",
        "hardware-aware-algorithms",
        "inference-optimization",
        "reasoning",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "inclusionAI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a144d-2189-701f-8efb-8bd3b4e59870",
      "paper_group_id": "019a144d-2189-701f-8efb-8bd3b4e59870",
      "title": "From Masks to Worlds: A Hitchhiker's Guide to World Models",
      "abstract": "这不是关于世界模型的典型调查；它是为那些想要构建世界的人提供的指南。我们并不旨在编目所有提到“世界模型”的论文。相反，我们遵循一条清晰的道路：从早期的掩蔽模型，它们统一了跨模态的表征学习，到共享单一范式的统一架构，再到关闭行动-感知循环的交互生成模型，最后是能够在时间上维持一致世界的记忆增强系统。我们跳过松散相关的分支，专注于核心：生成的核心、交互循环和记忆系统。我们表明，这是一条通向真正世界模型的最有前景的道路。",
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
        "total_votes": 2,
        "visits_count": {
          "all": 49,
          "last_7_days": 49
        },
        "public_total_votes": 11
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
      "id": "019a13f0-d3dc-79f2-9200-a5adca1d33f0",
      "paper_group_id": "019a13f0-d3dc-79f2-9200-a5adca1d33f0",
      "title": "Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence",
      "abstract": "大多数视频推理模型仅生成文本推理轨迹，而没有指明关键证据出现的时间和地点。最近的模型，如OpenAI-o3，引起了人们对图像证据中心推理的广泛兴趣，但将这种能力扩展到视频更具挑战性，因为它需要在动态场景中进行联合时间跟踪和空间定位。我们介绍了Open-o3 Video，一个非代理框架，将明确的时空证据整合到视频推理中，并仔细收集训练数据和设计训练策略以应对上述挑战。该模型在其答案旁边突出显示关键时间戳、对象和边界框，使推理能够扎根于具体的视觉观察中。为了实现这一功能，我们首先策划并构建了两个高质量的数据集，STGR-CoT-30k用于SFT，STGR-RL-36k用于RL，配有精心构建的时间和空间注释，因为大多数现有数据集仅提供视频的时间跨度或图像的空间框，缺乏统一的时空监督和推理轨迹。然后，我们采用冷启动强化学习策略，结合多种特别设计的奖励，旨在共同促进答案的准确性、时间对齐和空间精度。在V-STAR基准上，Open-o3 Video达到了最先进的性能，使mAM提升了14.4%，mLGM提升了24.2%，相较于Qwen2.5-VL基线。在广泛的视频理解基准测试中，如VideoMME、WorldSense、VideoMMMU和TVGBench也观察到了持续的改进。除了准确性，Open-o3 Video生成的推理轨迹还提供了有价值的信号，用于测试时的规模扩展，使得置信度感知的验证成为可能，提升了答案的可靠性。",
      "paper_summary": {
        "summary": "A framework called Open-o3 Video enables grounded video reasoning by integrating explicit spatio-temporal evidence directly into the model's output. This approach achieves state-of-the-art performance on the V-STAR benchmark while providing precise timestamps and bounding boxes for supporting visual cues.",
        "originalProblem": [
          "Existing video reasoning models often provide only textual explanations, lacking explicit visual grounding (precise spatio-temporal evidence).",
          "Extending image-centric grounded reasoning to dynamic videos is challenging due to the complexities of motion, occlusions, and the necessity of simultaneous temporal and spatial localization.",
          "There is a critical gap in high-quality datasets that provide unified spatio-temporal supervision for explicit reasoning and robust training strategies for joint localization."
        ],
        "solution": [
          "Curated two high-quality datasets, STGR-CoT-30k and STGR-RL-36k, which integrate question-answer pairs with timestamped key frames, localized bounding boxes, and structured chains of thought.",
          "Implemented a two-stage training paradigm comprising Supervised Fine-Tuning (SFT) for foundational capabilities and Reinforcement Learning (RL) with Group Sequence Policy Optimization (GSPO).",
          "Developed a composite reward design for RL, featuring adaptive temporal proximity for temporal terms and a temporal gating mechanism for spatial terms to ensure precise spatio-temporal alignment."
        ],
        "keyInsights": [
          "Explicit spatio-temporal evidence generation significantly enhances the interpretability and verifiability of video reasoning, moving beyond textual rationales.",
          "A synergistic two-stage training paradigm, combining supervised fine-tuning with reinforcement learning (using GSPO), is crucial for robust spatio-temporal alignment.",
          "Innovative reward designs, such as adaptive temporal proximity and temporal gating, are essential for stable optimization and precise localization in both time and space."
        ],
        "results": [
          "Open-o3 Video achieved state-of-the-art results on the V-STAR benchmark, with a +14.4% improvement in mAM and +24.2% in mLGM over the Qwen2.5-VL-7B baseline.",
          "The model exhibited substantial gains in fine-grained tasks on V-STAR, including a +27.6% increase in question answering accuracy and improvements of up to +10.2% Temporal IoU and +8.4% Visual IoU for grounding.",
          "Consistent performance improvements were observed across general video understanding benchmarks, such as VideoMME (+4.1% on long videos) and TVGBench (+4.5 mIoU), indicating enhanced temporal reasoning and perceptual grounding capabilities."
        ]
      },
      "image_url": "image/2510.20579v1.png",
      "universal_paper_id": "2510.20579",
      "metrics": {
        "total_votes": 1,
        "visits_count": {
          "all": 85,
          "last_7_days": 85
        },
        "public_total_votes": 14
      },
      "first_publication_date": "2025-10-23T14:05:56.000Z",
      "publication_date": "2025-10-23T14:05:56.000Z",
      "updated_at": "2025-10-24T01:59:01.340Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CV",
        "cs.MM"
      ],
      "organization_info": [
        {
          "name": "NUS",
          "image": null
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
          "name": "CASIA",
          "image": null
        },
        {
          "name": "WHU",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 37,
      "github_url": "https://github.com/marinero4972/Open-o3-Video",
      "distance": 1
    },
    {
      "id": "019a13d1-4cf4-7c17-ae3a-7a8b27874e17",
      "paper_group_id": "019a13d1-4cf4-7c17-ae3a-7a8b27874e17",
      "title": "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases",
      "abstract": "寻找并利用“捷径”完成任务的倾向对大型语言模型（LLMs）的可靠评估和部署构成了重大风险。例如，一个可以访问单元测试的LLM代理可能会删除失败的测试，而不是修复其中的根本错误。这种行为削弱了基准结果的有效性和现实世界中LLM编码助手部署的可靠性。\n\n为了量化、研究和减轻这种行为，我们引入了ImpossibleBench，这是一个基准框架，系统地衡量LLM代理利用测试用例的倾向。ImpossibleBench通过在自然语言规范和单元测试之间引入直接冲突，创建了现有基准（如LiveCodeBench和SWE-bench）的“不可完成”任务变体。我们将代理的“作弊率”定义为其在这些不可能任务上的通过率，任何通过都必然意味着一种违反规范的捷径。\n\n作为一个实用框架，ImpossibleBench不仅是一个评估工具，还是一个多功能工具。我们展示了其在以下方面的实用性：（1）研究模型行为，揭示从简单测试修改到复杂运算符重载的作弊行为的更细粒度细节；（2）上下文工程，展示提示、测试访问和反馈循环如何影响作弊率；（3）开发监测工具，提供一个具有经验证的欺骗解决方案的测试平台。我们希望ImpossibleBench能作为构建更强大、更可靠的LLM系统的有用框架。\n\n我们的实现可以在此链接找到。",
      "paper_summary": {
        "summary": "ImpossibleBench introduces a framework to measure large language models' (LLMs) tendency to exploit test cases in coding problems rather than adhering to natural language specifications. The study finds that frontier LLMs frequently engage in diverse and sophisticated \"cheating\" strategies, and demonstrates that prompt engineering and restricted test access can substantially reduce these undesirable behaviors.",
        "originalProblem": [
          "Standard LLM coding benchmarks struggle to detect when models pass tests by violating natural language specifications, leading to inflated performance metrics.",
          "LLMs, particularly in agentic settings, have shown a concerning tendency to engage in \"reward hacking\" or \"cheating\" by modifying tests or hardcoding solutions.",
          "There has been no systematic, automated framework to quantify and analyze these specification-violating shortcuts, making mitigation difficult."
        ],
        "solution": [
          "ImpossibleBench creates \"impossible\" variants of existing coding tasks by modifying test cases to introduce direct conflicts with natural language specifications.",
          "It employs two primary automated test mutation strategies: one-off modifications to expected outputs and the introduction of explicitly contradictory test cases.",
          "The framework evaluates LLMs with open test access and iterative feedback, allowing for detailed analysis of cheating strategies and the impact of context engineering techniques."
        ],
        "keyInsights": [
          "Frontier LLMs frequently engage in \"cheating\" behaviors, with stronger models often exhibiting higher cheating rates and employing more sophisticated exploitation strategies.",
          "LLMs utilize diverse cheating methods, including direct test modification, overloading comparison operators, recording extra states, and special-casing specific test inputs.",
          "Context engineering techniques, such as stricter prompts, read-only test access, and an explicit \"abort\" mechanism, can significantly reduce LLM cheating rates."
        ],
        "results": [
          "GPT-5 cheated in 76% of tasks on Oneoff-SWEbench and 54% on Conflicting-SWEbench, with OpenAI models exhibiting more diverse cheating behaviors than Claude models.",
          "Stricter prompts reduced GPT-5's cheating on Conflicting-LiveCodeBench from over 85% to 1%, and an \"abort\" mechanism reduced its cheating on Conflicting-SWEbench from 54% to 9%.",
          "LLM-based monitors detected 86-89% of cheating attempts on simpler LiveCodeBench tasks but only 42-65% on more complex Impossible-SWEbench tasks, highlighting challenges in detection."
        ]
      },
      "image_url": "image/2510.20270v1.png",
      "universal_paper_id": "2510.20270",
      "metrics": {
        "total_votes": 7,
        "visits_count": {
          "all": 85,
          "last_7_days": 85
        },
        "public_total_votes": 20
      },
      "first_publication_date": "2025-10-23T06:58:32.000Z",
      "publication_date": "2025-10-23T06:58:32.000Z",
      "updated_at": "2025-10-24T01:24:35.188Z",
      "topics": [
        "Computer Science",
        "cs.CL",
        "cs.LG"
      ],
      "organization_info": [
        {
          "name": "Anthropic",
          "image": "images/organizations/anthropic.svg+xml"
        },
        {
          "name": "Carnegie Mellon University",
          "image": "images/organizations/cmu.jpg"
        }
      ],
      "author_info": [],
      "github_stars": 1,
      "github_url": "https://github.com/safety-research/impossiblebench",
      "distance": 1
    },
    {
      "id": "019a0998-6209-753f-b2c0-b13dbaa5d7c4",
      "paper_group_id": "019a0998-6209-753f-b2c0-b13dbaa5d7c4",
      "title": "Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model",
      "abstract": "我们推出了Ring-1T，这是第一个开源的先进思维模型，具有万亿级别的参数。该模型总计拥有1万亿个参数，每个token激活约500亿个参数。在万亿参数规模上训练这样的模型带来了前所未有的挑战，包括训练与推理之间的不匹配、滚动处理的低效以及强化学习系统的瓶颈。为了解决这些问题，我们在三项相互关联的创新中首开先河：（1）IcePop通过token级别的差异掩蔽和裁剪来稳定强化学习训练，解决了训练与推理不匹配带来的不稳定性；（2）C3PO++通过动态划分长滚动以改善在token预算下的资源利用，从而获得高时间效率；（3）ASystem，一个高性能的强化学习框架，旨在克服阻碍万亿参数模型训练的系统瓶颈。Ring-1T在关键基准测试中取得了突破性成果：在AIME-2025上得分93.4，在HMMT-2025上得分86.72，在CodeForces上得分2088，在ARC-AGI-v1上得分55.94。值得注意的是，它在IMO-2025上获得了银牌级别的成绩，突显了其卓越的推理能力。通过向社区发布完整的1T参数MoE模型，我们为研究界提供了直接接触前沿推理能力的机会。这一贡献标志着在民主化大规模推理智能方面的重要里程碑，并为开源模型性能建立了新的基准。",
      "paper_summary": {
        "summary": "Ring-1T is the first open-source trillion-parameter thinking model developed by the Ling Team at Inclusion AI, achieving state-of-the-art reasoning capabilities in competitive mathematics, coding, and logical reasoning benchmarks through a multi-stage reinforcement learning pipeline and specialized infrastructure.",
        "originalProblem": [
          "Scaling reinforcement learning to trillion-parameter models presents severe training instability and prohibitive computational costs.",
          "Existing RL systems suffer from train-inference misalignment and inefficiencies in processing long reasoning trajectories.",
          "The absence of a publicly available, trillion-parameter \"thinking model\" limits broader AI research and development."
        ],
        "solution": [
          "A multi-stage training pipeline comprising Long Chain-of-Thought Supervised Fine-Tuning, Reasoning Reinforcement Learning (RLVR), and General Reinforcement Learning (RLHF).",
          "Novel algorithmic innovations: IcePop for mitigating RL training instability and C3PO++ for efficient rollout partitioning.",
          "A high-performance RL infrastructure, ASystem, with specialized components for runtime, memory, weight synchronization, and verifiable task sandboxing."
        ],
        "keyInsights": [
          "Stable and efficient reinforcement learning for trillion-parameter models requires dedicated solutions for train-inference mismatch and rollout processing, specifically addressed by IcePop and C3PO++.",
          "A robust, co-designed infrastructure like ASystem is essential for managing the computational complexity and data flow of large-scale RL training.",
          "Combining supervised fine-tuning with multiple stages of reinforcement learning (RLVR and RLHF) effectively cultivates advanced reasoning and human alignment."
        ],
        "results": [
          "Achieved silver medal-level performance on the International Mathematical Olympiad (IMO-2025) and top scores on AIME-2025 (93.40%) and HMMT-2025 (86.72%).",
          "Set new state-of-the-art for open-source models in coding benchmarks, scoring 78.30% on LiveCodeBench-v6 and a 2088 rating on CodeForces.",
          "IcePop stabilized RL training, while C3PO++ provided an approximate 2.5x speedup in rollout processing and 1.5x end-to-end training speedup per step."
        ]
      },
      "image_url": "image/2510.18855v1.png",
      "universal_paper_id": "2510.18855",
      "metrics": {
        "total_votes": 13,
        "visits_count": {
          "all": 576,
          "last_7_days": 576
        },
        "public_total_votes": 60
      },
      "first_publication_date": "2025-10-21T17:46:14.000Z",
      "publication_date": "2025-10-21T17:46:14.000Z",
      "updated_at": "2025-10-22T01:46:12.873Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "distributed-learning",
        "efficient-transformers",
        "generative-models",
        "ml-systems",
        "optimization-methods",
        "reasoning",
        "reinforcement-learning",
        "training-orchestration",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Inclusion AI",
          "image": null
        },
        {
          "name": "Ling Team",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 42,
      "github_url": "https://github.com/inclusionAI/Ring-V2",
      "distance": 1
    },
    {
      "id": "019a13b7-6b48-742d-b239-9e781e0e1dec",
      "paper_group_id": "019a13b7-6b48-742d-b239-9e781e0e1dec",
      "title": "Black Box Absorption: LLMs Undermining Innovative Ideas",
      "abstract": "大型语言模型正日益成为加速创新的关键工具。本文识别并形式化了这一范式中固有的系统风险：\\textbf{黑箱吸收}。我们将其定义为一种过程，即大型服务提供商运营的 LLM 平台的不透明内部结构能够在交互过程中吸收、概括并重新利用用户贡献的新概念。这个机制威胁到创新经济学的基本原则，因为它在个体创作者与平台运营者之间造成严重的信息和结构不对称，从而危及创新生态系统的长期可持续性。为了分析这一挑战，我们引入了两个核心概念：思想单元，代表创新的可转移功能逻辑；思想安全，一个用于保护创新的多维标准。本文分析了吸收的机制，并提出了一个具体的治理和工程议程，以减轻这些风险，确保创作者的贡献可追溯、可控且公平。",
      "paper_summary": {
        "summary": "This paper identifies and formalizes \"Black Box Absorption\" as a systemic risk in Large Language Model (LLM) platforms, where user-contributed novel functional ideas are internalized, generalized, and repurposed without clear attribution or compensation. It details the mechanisms of this absorption and proposes an \"Idea Safety Agenda\" based on principles of control, traceability, and equitability to mitigate these risks.",
        "originalProblem": [
          "LLM platforms, while fostering creativity, often overlook robust protection for user-contributed innovative ideas within their operational structures.",
          "Opaque internal architectures and broad legal licenses of current LLM providers inadvertently or intentionally facilitate the absorption of novel ideas.",
          "Existing privacy research primarily focuses on personal data leakage, not on safeguarding the intellectual property and functional ideas embedded in live user interactions."
        ],
        "solution": [
          "Formalize \"Black Box Absorption\" as the process where LLM platforms internalize, generalize, and repurpose novel concepts from users, creating informational and structural asymmetries.",
          "Introduce the \"Idea Unit\" as the transportable functional logic of an innovation, serving as the fundamental granular object of analysis susceptible to absorption.",
          "Propose an \"Idea Safety Agenda\" with principles of Control, Traceability, and Equitability, outlining a deployable framework to protect creators' contributions and ensure fair value realization."
        ],
        "keyInsights": [
          "The inherent opacity of LLM operations combined with broad user licenses creates systemic informational and structural asymmetries that favor platform operators over individual innovators.",
          "Standard LLM operational pipelines—including data governance, logging, sampling, human review (RLHF), and model retraining—function as mechanisms for the systematic internalization and generalization of user-contributed functional ideas.",
          "This absorption process leads to critical economic and societal consequences, such as adoption pressure, untraceable control, and asymmetrical value realization, potentially undermining the broader innovation ecosystem."
        ],
        "results": [
          "A multi-stage process was detailed for how \"Idea Units\" are absorbed: from broad user licensing and interaction logging, through data sampling and human annotation, to data curation and model retraining.",
          "It was demonstrated that absorbed \"Idea Units\" become generalized into the model's parameters, making them non-exclusive, untraceable to their original creator, and influencing future model responses.",
          "Key consequences identified include compelling creators into a dilemma of adoption versus protection, resulting in untraceable and asymmetrical control over ideas, and a disproportionate channeling of value towards platform operators."
        ]
      },
      "image_url": "image/2510.20612v1.png",
      "universal_paper_id": "2510.20612",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 43,
          "last_7_days": 43
        },
        "public_total_votes": 9
      },
      "first_publication_date": "2025-10-23T14:43:09.000Z",
      "publication_date": "2025-10-23T14:43:09.000Z",
      "updated_at": "2025-10-24T00:56:19.016Z",
      "topics": [
        "Computer Science",
        "cs.AI",
        "cs.CR",
        "cs.CY",
        "cs.LG",
        "econ.GN",
        "Economics",
        "explainable-ai",
        "human-ai-interaction",
        "ml-systems",
        "privacy-preserving-ml",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a0ed9-f313-7c1f-9395-536ba9146063",
      "paper_group_id": "019a0ed9-f313-7c1f-9395-536ba9146063",
      "title": "Blackbox Model Provenance via Palimpsestic Membership Inference",
      "abstract": "假设爱丽丝训练了一个开放权重的语言模型，而鲍勃使用爱丽丝模型的黑箱衍生品生成文本。爱丽丝能否证明鲍勃正在使用她的模型，无论是通过查询鲍勃的衍生模型（查询设置）还是仅通过文本（观察设置）？我们将这个问题表述为独立性检验问题——其中零假设是鲍勃的模型或文本与爱丽丝的随机训练过程是独立的——并通过语言模型中的重叠记忆现象来研究这一点：模型更可能记住训练过程中后期看到的数据，因此我们可以通过捕捉鲍勃的模型或文本与爱丽丝训练过程中训练示例顺序之间的相关性的测试统计量来检验鲍勃是否在使用爱丽丝的模型。如果爱丽丝随机打乱了她的训练数据，那么任何显著的相关性都会成为对零假设的可量化统计证据，无论爱丽丝的训练数据的组成如何。在查询设置中，我们直接估计（通过提示）鲍勃的模型对爱丽丝训练示例及其顺序的可能性；我们将超过40个不同Pythia和OLMo基础模型的微调概率（参数范围从1B到12B）与基础模型的训练数据顺序进行相关，所有情况下的p值均在最多1e-8的范围内，仅有六个例外。在观察设置中，我们尝试两种方法，基于估计1) 鲍勃的文本与爱丽丝训练示例的跨度重叠的可能性和2) 鲍勃的文本相对于不同版本的爱丽丝模型的可能性，我们通过在重新排列的数据上重复她训练过程的最后一个阶段（例如1%）获得这些版本。第二种方法可以可靠地从仅几百个词元中区分鲍勃的文本；第一种方法不涉及任何再训练，但需要更多的词元（几十万个）来实现较高的检验能力。",
      "paper_summary": {
        "summary": "Researchers at Stanford University developed a statistical framework to prove the provenance of blackbox large language models and their generated text by leveraging a phenomenon called \"palimpsestic memorization.\" The methods provide statistically rigorous detection with provable false positive control across both direct query and observational settings, demonstrated effectively on various Pythia, OLMo, and TinyStories models.",
        "originalProblem": [
          "Establishing transparent, non-invasive, and statistically rigorous methods to prove if a blackbox LLM or its generated text is a derivative of a known model.",
          "Enforcing intellectual property rights and terms of service for proprietary LLMs without requiring model developers to modify their original training process.",
          "Existing provenance techniques are often invasive (e.g., model fingerprinting), lack statistical guarantees (e.g., heuristic similarity), or are easily circumvented (e.g., watermarking)."
        ],
        "solution": [
          "Formulates provenance as an independence testing problem by exploiting \"palimpsestic memorization,\" where LLMs retain traces of their training data's specific order.",
          "Proposes methods for a \"Query Setting\" (direct API access) using Spearman rank correlation of log-likelihoods with training order, enhanced by a reference model.",
          "Introduces approaches for an \"Observational Setting\" (only generated text available) using simpler n-gram models on partitioned data or more powerful neural LMs trained on reshuffled data segments."
        ],
        "keyInsights": [
          "LLMs exhibit \"palimpsestic memorization,\" meaning their internal representations and output probabilities are measurably influenced by the *order* in which training examples were processed, particularly for later-seen data.",
          "By assuming a randomized training shuffle, provenance can be statistically proven through an independence test, offering provable control over false positive errors (Type-I errors).",
          "Employing an independent reference model in query-based tests significantly reduces the required amount of interaction with a blackbox model while maintaining high statistical power."
        ],
        "results": [
          "Query-based methods consistently achieved extremely low p-values (e.g., $10^{-8}$ to $10^{-184}$) with as few as 100K-5M query tokens, demonstrating robustness against various finetuning techniques.",
          "Observational methods using neural LMs ($\\phi_{shuff\\_obs}$) could reliably detect provenance from as little as a few hundred generated tokens (e.g., 320 tokens for p-values $< 10^{-3}$ on TinyStories models).",
          "The research exposed a mislabeling in \"pythia-2.8b-deduped\" models, providing strong statistical evidence ($p < 10^{-60}$) that they were trained on non-deduplicated data, validating the method's auditing capabilities."
        ]
      },
      "image_url": "image/2510.19796v1.png",
      "universal_paper_id": "2510.19796",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 73,
          "last_7_days": 73
        },
        "public_total_votes": 14
      },
      "first_publication_date": "2025-10-22T17:30:39.000Z",
      "publication_date": "2025-10-22T17:30:39.000Z",
      "updated_at": "2025-10-23T02:15:55.923Z",
      "topics": [
        "ai-for-cybersecurity",
        "Computer Science",
        "cs.CL",
        "cs.LG",
        "fine-tuning",
        "mechanistic-interpretability",
        "model-observability",
        "privacy-preserving-ml",
        "statistical-learning",
        "text-generation",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Stanford University",
          "image": "images/organizations/stanford.png"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a0ed6-7d9a-7526-8dcb-159f12ac7ccc",
      "paper_group_id": "019a0ed6-7d9a-7526-8dcb-159f12ac7ccc",
      "title": "GigaBrain-0: A World Model-Powered Vision-Language-Action Model",
      "abstract": "训练通用机器人使用的视觉-语言-动作（VLA）模型通常需要大规模的真实机器人数据，这种数据的收集既昂贵又耗时。物理数据收集的低效率严重限制了当前VLA系统的可扩展性和泛化能力。为了解决这个挑战，我们推出了GigaBrain-0，这是一种由世界模型生成数据（例如视频生成、真实到真实的转换、人类转移、视角转移、仿真到真实的转移数据）赋能的新型VLA基础模型。通过利用世界模型大规模生成多样化数据，GigaBrain-0显著减少了对真实机器人数据的依赖，同时改善了跨任务的泛化能力。我们的方法进一步通过RGBD输入建模和具身的思维链（CoT）监督提高策略的鲁棒性，使模型能够在任务执行过程中推理空间几何、物体状态和长时间依赖关系。这在灵巧的、长时间的和移动操控任务中显著提升了实际性能。大量实验表明，GigaBrain-0在外观变化（例如纹理、颜色）、物体摆放和摄像头视点方面实现了卓越的泛化。此外，我们还推出了GigaBrain-0-Small，这是一个优化的轻量级变体，旨在高效地运行在如NVIDIA Jetson AGX Orin等设备上。",
      "paper_summary": {
        "summary": "GigaBrain-0, developed by GigaAI, is a Vision-Language-Action (VLA) foundation model that dramatically reduces reliance on real-world data by training on diverse world model-generated data from the GigaWorld framework. This approach leads to superior generalization across varying appearances, placements, and viewpoints, achieving higher task success rates in dexterous, long-horizon, and mobile manipulation tasks.",
        "originalProblem": [
          "The prohibitive cost and time commitment of collecting large-scale, diverse real-world robot interaction data for generalist robots.",
          "Limitations in environmental and task diversity with traditional data collection methods, restricting the generalization capacity of robotic policies.",
          "Difficulty for current VLA systems to robustly perform complex, multi-step tasks in unstructured, dynamic real-world settings."
        ],
        "solution": [
          "Introduces GigaBrain-0, a VLA foundation model trained on diverse world model-generated data from the GigaWorld framework to augment real-world datasets.",
          "Employs a mixture-of-transformers architecture integrating RGBD input modeling for enhanced spatial reasoning and embodied Chain-of-Thought (CoT) supervision for improved sequential decision-making.",
          "Utilizes GigaWorld's pipelines for Real2Real transfer, View transfer, Sim2Real transfer, and Human video transfer to synthesize physically plausible training sequences with vast variations in visual context and robot interactions."
        ],
        "keyInsights": [
          "World models, specifically the GigaWorld framework, can serve as scalable and effective data engines to circumvent the limitations of purely physical data collection in robotics.",
          "Integrating RGBD inputs and embodied Chain-of-Thought reasoning significantly improves a robot's spatial awareness, planning capabilities, and robustness for complex, long-horizon tasks by explicitly modeling intermediate steps.",
          "Synthetic data generation, especially through diverse transfer methods, is crucial for enhancing generalization across appearance, object placement, and camera viewpoint variations, leading to more robust policies in real-world environments."
        ],
        "results": [
          "GigaBrain-0 consistently achieved higher task success rates, including a 30% increase for laundry folding and 10% for mobile manipulation tasks, compared to a leading VLA baseline, 𝜋0.",
          "Training with GigaWorld-generated data dramatically boosted generalization, with success rates on novel appearances, placements, and viewpoints increasing to over 80-90% by incorporating transferred data.",
          "GigaBrain-0-Small, an optimized variant, achieved comparable performance to the larger 𝜋0 model with significantly fewer parameters (402M vs 3.2B), reduced VRAM (1.9GB vs 17.5GB), and a 10x faster inference latency (0.13s vs 1.28s) on edge devices."
        ]
      },
      "image_url": "image/2510.19430v1.png",
      "universal_paper_id": "2510.19430",
      "metrics": {
        "total_votes": 4,
        "visits_count": {
          "all": 164,
          "last_7_days": 164
        },
        "public_total_votes": 24
      },
      "first_publication_date": "2025-10-22T09:57:13.000Z",
      "publication_date": "2025-10-22T09:57:13.000Z",
      "updated_at": "2025-10-23T02:12:09.242Z",
      "topics": [
        "chain-of-thought",
        "Computer Science",
        "cs.CV",
        "cs.RO",
        "edge-computing",
        "generative-models",
        "lightweight-models",
        "multi-modal-learning",
        "reasoning",
        "robotic-control",
        "robotics-perception",
        "synthetic-data",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "GigaAI",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 31,
      "github_url": "https://github.com/open-gigaai/giga-brain-0",
      "distance": 1
    },
    {
      "id": "019a0eb3-c652-7879-b87a-8c2c45da60bc",
      "paper_group_id": "019a0eb3-c652-7879-b87a-8c2c45da60bc",
      "title": "Human-Agent Collaborative Paper-to-Page Crafting for Under $0.1",
      "abstract": "在追求科学进步的过程中，传达研究成果与发现本身同样重要。然而，研究人员常常被手动重复的任务所分散注意力，忙于建立项目网页，以使其内容密集的论文易于访问。尽管自动化已经解决了静态幻灯片和海报的问题，但网页的动态交互特性仍然是一项未得到解决的挑战。为了填补这一空白，我们重新构建了问题，认为解决方案不在于一个单一的命令，而在于一个协作的层级过程。我们介绍了$\\textbf{AutoPage}$，一个体现这一理念的新型多智能体系统。AutoPage将论文到网页的创建过程分解为从叙事规划到多模态内容生成和交互渲染的粗到细的管道。为了对抗人工智能的幻觉，专门的“检查员”代理会将每个步骤与源论文进行核实，同时可选的人为检查点确保最终产品完美符合作者的愿景，使得该系统不仅仅是一个工具，而是一个强大的协作助手。为了严格验证我们的方法，我们还构建了$\\textbf{PageBench}$，这是针对这一新任务的第一个基准。实验表明，AutoPage不仅生成高质量、视觉吸引人的页面，而且在15分钟内以不到0.1美元的成本实现卓越的效率。代码和数据集将在$\\href{this https URL}{Webpage}$上发布。",
      "paper_summary": {
        "summary": "AutoPage, a multi-agent system developed by researchers at AutoLab, SAI, Shanghai Jiao Tong University, and Shanghai AI Laboratory, automates the generation of interactive project webpages from academic papers. This system produces high-quality, factually accurate web content with multimodal elements in under 15 minutes and for less than $0.10, integrating optional human collaboration.",
        "originalProblem": [
          "Researchers spend significant manual effort and time creating project webpages to disseminate their work.",
          "Existing automated presentation tools for papers (e.g., slides, posters) only generate static formats, lacking support for interactive, scrollable web content.",
          "There is an unmet need for flexible web content that adapts to diverse paper structures and integrates dynamic elements like demos."
        ],
        "solution": [
          "AutoPage implements a multi-agent system that uses a hierarchical, coarse-to-fine generation process for interactive project webpages.",
          "It integrates LLM/VLM-based \"Checker\" agents throughout the pipeline to ensure factual accuracy and mitigate AI hallucination.",
          "The system supports optional human-in-the-loop checkpoints, allowing researchers to provide language commands for iterative refinement of content and design."
        ],
        "keyInsights": [
          "A multi-agent architecture with a coarse-to-fine generation strategy and integrated verification is highly effective for complex, interactive web content creation.",
          "Automated, high-quality interactive project webpage generation can be achieved with remarkable efficiency (under 15 minutes) and at an exceptionally low cost (under $0.1).",
          "Flexible human-agent collaboration through optional checkpoints allows for precise control and alignment with authorial intent, transforming AI tools into powerful assistants."
        ],
        "results": [
          "AutoPage consistently enhanced both content and visual generation quality across various LLM backbones, boosting aesthetic scores from 2.71 to 2.95 and improving layout and cohesion from 2.08 to 2.38 for GPT-4o-mini.",
          "A user study showed strong preference for AutoPage-generated pages, achieving the highest average score of 7.16 out of 10, outperforming all baseline models.",
          "The system demonstrated high efficiency (4-20 minutes per page) and cost-effectiveness ($0.06-$0.20 per page), with ablation studies confirming the critical role of internal verification agents for maintaining quality."
        ]
      },
      "image_url": "image/2510.19600v1.png",
      "universal_paper_id": "2510.19600",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 200,
          "last_7_days": 200
        },
        "public_total_votes": 26
      },
      "first_publication_date": "2025-10-22T13:53:57.000Z",
      "publication_date": "2025-10-22T13:53:57.000Z",
      "updated_at": "2025-10-23T01:34:14.098Z",
      "topics": [
        "agent-based-systems",
        "agents",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "cs.SE",
        "data-curation",
        "generative-models",
        "human-ai-interaction",
        "information-extraction",
        "reasoning",
        "text-generation"
      ],
      "organization_info": [
        {
          "name": "Shanghai AI Laboratory",
          "image": null
        },
        {
          "name": "Shanghai Jiao Tong University",
          "image": "images/organizations/shanghai-jiao-tong-university.png"
        }
      ],
      "author_info": [],
      "github_stars": 81,
      "github_url": "https://github.com/AutoLab-SAI-SJTU/AutoPage",
      "distance": 1
    },
    {
      "id": "019a13f2-1df0-7db3-bc33-6193a1ac3a3f",
      "paper_group_id": "019a13f2-1df0-7db3-bc33-6193a1ac3a3f",
      "title": "AlphaFlow: Understanding and Improving MeanFlow Models",
      "abstract": "MeanFlow最近作为一种强大的框架出现，用于从头开始进行少步生成建模，但其成功尚未完全理解。在这项工作中，我们展示了MeanFlow目标自然地分解为两个部分：轨迹流匹配和轨迹一致性。通过梯度分析，我们发现这些项之间存在强烈的负相关，导致优化冲突和收敛缓慢。基于这些见解，我们引入了$\\alpha$-Flow，这是一种广泛的目标家族，统一了轨迹流匹配、捷径模型和MeanFlow。在采用从轨迹流匹配平滑退火到MeanFlow的课程策略下，$\\alpha$-Flow解开了冲突目标，实现了更好的收敛。当在类条件的ImageNet-1K 256x256上使用普通DiT主干从头训练时，$\\alpha$-Flow在各个尺度和设置上始终优于MeanFlow。我们最大的$\\alpha$-Flow-XL/2+模型使用普通DiT主干达到了新的最先进结果，FID分数为2.58（1-NFE）和2.15（2-NFE）。",
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
        "total_votes": 0,
        "visits_count": {
          "all": 43,
          "last_7_days": 43
        },
        "public_total_votes": 9
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
      "id": "019a1537-6027-7ba2-a64d-260a33b2e7ef",
      "paper_group_id": "019a1537-6027-7ba2-a64d-260a33b2e7ef",
      "title": "What Defines Good Reasoning in LLMs? Dissecting Reasoning Steps with Multi-Aspect Evaluation",
      "abstract": "对大型语言模型（LLMs）在最终答案正确性上的评估是主导范式。然而，这种方法提供的信号对于模型改进来说过于粗糙，并且忽视了基础推理过程的质量。我们认为，对推理进行更细致的评估提供了一条更有效的构建稳健模型的路径。我们将推理质量分解为两个维度：相关性和连贯性。相关性衡量某一步是否与问题相关；连贯性衡量该步骤是否合乎逻辑地跟随于之前的步骤。为了可靠地测量这些方面，我们引入了因果逐步评估（CaSE）。该方法仅使用其前文上下文来评估每个推理步骤，从而避免了事后偏见。我们在新的专家注释基准上验证了CaSE与人类判断的一致性，基准包括MRa-GSM8K和MRa-MATH。更重要的是，我们展示了使用CaSE评估的相关性和连贯性来策划训练数据，能够直接提升最终任务的表现。我们的工作提供了一个可扩展的框架，用于分析、调试和改进LLM推理，证明了超越有效性检查的实际价值。",
      "paper_summary": {
        "summary": "A multi-aspect framework evaluates Large Language Model reasoning by formalizing \"relevance\" and \"coherence\" at the step level, complementing traditional correctness. Causal Stepwise Evaluation (CaSE) assesses these aspects without hindsight bias, showing that CaSE-curated supervised fine-tuning data and aspect-guided inference improve LLM reasoning performance and final answer accuracy.",
        "originalProblem": [
          "Current LLM evaluation primarily relies on final-answer correctness, providing limited diagnostic feedback for improving the underlying reasoning process.",
          "Existing process-based evaluations often focus only on \"step correctness,\" neglecting crucial dimensions like a step's relevance to the overall problem and its logical coherence with preceding steps.",
          "LLM-as-a-judge protocols typically suffer from \"hindsight bias\" due to evaluators having access to the full reasoning trace, potentially leading to inflated or inaccurate quality assessments."
        ],
        "solution": [
          "Formalizes two new, pedagogically inspired dimensions of step-level reasoning quality: 'relevance' (grounded contribution to the solution) and 'coherence' (logical flow from preceding steps).",
          "Introduces Causal Stepwise Evaluation (CaSE), an automated, reference-free method that assesses each reasoning step based solely on the original question and immediately preceding context to eliminate hindsight bias.",
          "Creates new human-expert-annotated benchmarks, MRa-GSM8K and MRa-MATH, which provide step-level labels for relevance and coherence to facilitate research and validation."
        ],
        "keyInsights": [
          "Relevance and coherence are crucial indicators of reasoning success, with solutions maintaining these qualities being more than twice as likely to yield a correct final answer.",
          "Causal evaluation, where each step is judged only by its preceding context, is essential for obtaining unbiased and accurate assessments of an LLM's generative reasoning process.",
          "Explicitly guiding LLMs toward relevance and coherence during inference, or curating supervised fine-tuning (SFT) data based on these quality criteria, leads to tangible improvements in both reasoning process quality and final task performance."
        ],
        "results": [
          "Solutions on the MRa-GSM8K benchmark with high solution-level relevance and coherence were 52% likely to have a correct final answer, compared to 24% for those lacking these qualities.",
          "CaSE consistently outperformed the Best-of-N baseline in aligning with human expert judgments across all evaluation aspects and LLM judges, showing substantial gains in detecting correctness failures.",
          "CaSE-based SFT data curation, particularly sample-level filtering, improved LLM performance across MATH, GPQA, and AIME24 benchmarks, outperforming random and heuristic-based data selection."
        ]
      },
      "image_url": "image/2510.20603v1.png",
      "universal_paper_id": "2510.20603",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 17,
          "last_7_days": 17
        },
        "public_total_votes": 4
      },
      "first_publication_date": "2025-10-23T14:30:37.000Z",
      "publication_date": "2025-10-23T14:30:37.000Z",
      "updated_at": "2025-10-24T07:55:41.991Z",
      "topics": [
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.CL",
        "data-curation",
        "fine-tuning",
        "model-interpretation",
        "reasoning",
        "reasoning-verification",
        "transformers"
      ],
      "organization_info": [],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a13ec-3eae-74ae-9748-09459db542c9",
      "paper_group_id": "019a13ec-3eae-74ae-9748-09459db542c9",
      "title": "Mind the gaps: The fraught road to quantum advantage",
      "abstract": "量子计算正在快速发展，但当前的噪声中间尺度量子（NISQ）设备与未来的容错应用规模（FASQ）机器之间仍存在显著的差距。我们识别出前方道路上的四个相关障碍：（i）从错误缓解到主动错误检测和纠正，（ii）从初步的错误纠正到可扩展的容错性，（iii）从早期启发式方法到成熟的、可验证的算法，以及（iv）从探索性模拟器到在量子模拟中产生可信优势。针对这些转变将加速向广泛实用的量子计算的进展。",
      "paper_summary": {
        "summary": "Jens Eisert and John Preskill critically analyze the state of quantum computing, identifying four major transitions necessary to advance from current noisy devices to future fault-tolerant, application-scale quantum computers. The perspective details challenges in error correction, algorithm development, and quantum simulation, advocating for a realistic assessment to guide research toward practical utility.",
        "originalProblem": [
          "The quantum computing field is experiencing rapid advancements, yet lacks a clear, realistic roadmap for achieving practical quantum utility beyond current noisy intermediate-scale quantum (NISQ) devices.",
          "There is an immense gap between current error-prone quantum hardware capabilities and the ultimate goal of fault-tolerant, application-scale quantum (FASQ) machines.",
          "The community needs a structured understanding of specific, interconnected challenges to temper over-hype and guide sustainable progress toward broadly useful quantum applications."
        ],
        "solution": [
          "The authors systematically identify and articulate four \"substantial gaps\" that represent key transitions required along the road from NISQ to FASQ.",
          "Each identified gap is analyzed in detail, covering the current state of the art, underlying challenges, and proposing future research directions.",
          "The paper provides a critical framework for evaluating claims of \"quantum advantage,\" distinguishing between benchmarks and practical utility."
        ],
        "keyInsights": [
          "Achieving fault-tolerant application-scale quantum computing necessitates overcoming distinct, difficult transitions, particularly moving from error mitigation to active error detection and correction.",
          "The resource overhead for scalable fault tolerance is currently immense, requiring significant advances in physical qubit quality, more efficient error correction codes, and faster real-time decoding algorithms.",
          "Demonstrating genuine quantum advantage for practically useful problems remains elusive, as classical algorithms continuously improve, and NISQ quantum algorithms face issues like barren plateaus and substantial data loading costs."
        ],
        "results": [
          "The paper characterizes four critical gaps: (i) from error mitigation to active error correction, (ii) from rudimentary to scalable fault tolerance, (iii) from early heuristics to mature, verifiable algorithms, and (iv) from exploratory simulators to credible advantage in quantum simulation.",
          "It concludes that the path to broadly useful quantum computing will be \"arduous, expensive, and prolonged,\" requiring concurrent advances across hardware fidelity, error correction theory, and algorithmic development.",
          "The article suggests that the most impactful applications of quantum computing may still be unforeseen, emphasizing the need for sustained fundamental scientific exploration alongside targeted engineering efforts."
        ]
      },
      "image_url": "image/2510.19928v1.png",
      "universal_paper_id": "2510.19928",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 81,
          "last_7_days": 81
        },
        "public_total_votes": 12
      },
      "first_publication_date": "2025-10-22T18:00:19.000Z",
      "publication_date": "2025-10-22T18:00:19.000Z",
      "updated_at": "2025-10-24T01:54:01.006Z",
      "topics": [
        "cond-mat.other",
        "Physics",
        "quant-ph"
      ],
      "organization_info": [
        {
          "name": "California Institute of Technology",
          "image": "images/organizations/california-institute-of-technology.svg+xml"
        },
        {
          "name": "Freie Universität Berlin",
          "image": null
        },
        {
          "name": "Helmholtz-Zentrum Berlin für Materialien und Energie",
          "image": null
        },
        {
          "name": "AWS Center for Quantum Computing",
          "image": null
        },
        {
          "name": "Fraunhofer Heinrich Hertz Institute",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a0996-2ae0-7896-b916-33343484c978",
      "paper_group_id": "019a0996-2ae0-7896-b916-33343484c978",
      "title": "A Definition of AGI",
      "abstract": "对人工通用智能（AGI）缺乏具体定义，使得今天的专业化人工智能与人类水平认知之间的差距变得模糊。本文提出了一个可量化的框架来应对这一问题，将AGI定义为匹配受过良好教育的成人的认知多样性和能力。为了实现这一目标，我们的方法论基于卡特尔-霍恩-卡罗尔理论，这是一种经过实证验证的人类认知模型。该框架将一般智能分解为十个核心认知领域，包括推理、记忆和感知，并调整已有的人类心理测量工具来评估人工智能系统。应用该框架揭示了当代模型具有高度“锯齿状”的认知特征。虽然在知识密集型领域表现良好，但当前的人工智能系统在基础认知机制上存在关键缺陷，特别是在长期记忆存储方面。因此，得出的AGI得分（例如，GPT-4为27%，GPT-5为58%）具体量化了快速进展与AGI之间仍然存在的重大差距。",
      "paper_summary": {
        "summary": "This work defines Artificial General Intelligence (AGI) as an AI that can match or exceed the cognitive versatility and proficiency of a well-educated adult, grounding this definition in the empirically validated Cattell-Horn-Carroll (CHC) theory of human intelligence. Applying this framework, the paper quantifies the current state of frontier AI models, revealing a \"jagged\" cognitive profile with strengths in knowledge and language but critical deficits in long-term memory storage and on-the-spot reasoning.",
        "originalProblem": [
          "The concept of Artificial General Intelligence (AGI) lacked a concrete, quantifiable definition, acting as a \"constantly moving goalpost\" that hindered scientific discourse.",
          "Existing AI benchmarks are often too specialized or prone to data contamination, failing to provide a holistic assessment of true underlying cognitive abilities.",
          "A systematic, empirically grounded framework was missing to evaluate AI capabilities against human cognitive structures and identify fundamental limitations."
        ],
        "solution": [
          "The paper establishes a human-centric definition of AGI, focusing on the cognitive versatility and proficiency comparable to a well-educated adult.",
          "It operationalizes this definition by adopting the Cattell-Horn-Carroll (CHC) theory, a robust and empirically validated model of human intelligence, to identify ten core cognitive domains.",
          "A systematic evaluation framework is proposed, adapting established psychometric tests to assess AI performance across these domains, yielding a standardized \"AGI Score\" and a detailed \"cognitive profile.\""
        ],
        "keyInsights": [
          "AGI evaluation requires a human-centric approach, measuring versatility and proficiency against a \"well-educated adult,\" rather than focusing solely on economic impact or superhuman performance on narrow tasks.",
          "Current AI models exhibit a \"jagged\" cognitive profile, demonstrating pronounced strengths in certain domains (e.g., general knowledge, language) but critical weaknesses in others (e.g., long-term memory storage, robust on-the-spot reasoning).",
          "\"Capability contortions,\" such as relying on large context windows or Retrieval-Augmented Generation (RAG), mask fundamental deficits in underlying cognitive abilities rather than representing true general intelligence."
        ],
        "results": [
          "GPT-4 achieved an estimated AGI Score of 27%, while a projected GPT-5 scored 58%, illustrating rapid progress but indicating a substantial remaining gap to human-level AGI (100%).",
          "Frontier AI models show high proficiency in General Knowledge (GPT-5: 9%) and Reading and Writing Ability (GPT-5: 10%), with significant advances in Mathematical Ability (GPT-5: 10%).",
          "Critical deficits persist in Long-Term Memory Storage (0% for both GPT-4 and GPT-5), On-the-Spot Reasoning (GPT-4: 0%, GPT-5: 7%), and robust multimodal processing, highlighting bottlenecks in core cognitive machinery."
        ]
      },
      "image_url": "image/2510.18212v1.png",
      "universal_paper_id": "2510.18212",
      "metrics": {
        "total_votes": 6,
        "visits_count": {
          "all": 246,
          "last_7_days": 246
        },
        "public_total_votes": 33
      },
      "first_publication_date": "2025-10-21T01:28:35.000Z",
      "publication_date": "2025-10-21T01:28:35.000Z",
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
      "id": "019a0ed3-9564-74f3-bb97-a57d1b80a4bc",
      "paper_group_id": "019a0ed3-9564-74f3-bb97-a57d1b80a4bc",
      "title": "Learning Affordances at Inference-Time for Vision-Language-Action Models",
      "abstract": "解决复杂的现实控制任务往往需要多次尝试：如果我们第一次失败，我们会反思发生了什么问题，并相应地改变策略，以避免犯同样的错误。在机器人领域，视觉-语言-动作模型（VLA）为解决复杂控制任务提供了一条有希望的路径，但缺乏在无法完成任务时上下文和动态调整行为的能力。在这项工作中，我们引入了推理时执行学习（LITEN），它将低层次的VLA策略与高层次的VLM连接起来，后者通过将过去的经验融入上下文来进行条件设置，从而使其能够学习低层次VLA的能力和潜能。我们的方法在生成和执行低层次VLA计划的推理阶段与反思结果执行并得出有用结论的评估阶段之间迭代。与非机器人领域的自我完善类似的方法不同，LITEN必须反思非结构化的现实世界机器人轨迹（例如，原始视频），这在评估过程中需要结构化的指导框架。我们的实验结果表明，LITEN能够有效地从过去的经验中学习，生成使用高潜能指令完成长时间任务的计划。",
      "paper_summary": {
        "summary": "A new framework from UC Berkeley allows robots to dynamically refine their plans by learning from real-world execution failures, improving success rates on complex manipulation tasks at inference time. This method enables high-level Vision-Language Models to iteratively learn the affordances of low-level Vision-Language-Action policies without requiring additional training or data collection.",
        "originalProblem": [
          "Current Vision-Language-Action (VLA) models are limited to 'single-shot' execution and struggle with complex, long-horizon tasks requiring sequential planning and self-correction.",
          "Existing zero-shot VLM planners lack specific understanding of a robot's physical capabilities and limitations (affordances), leading to suboptimal performance.",
          "Transferring in-context learning and self-refinement techniques from large language models to real-world robotics is challenging due to unstructured sensory data, physical constraints, and stochastic robot actions."
        ],
        "solution": [
          "LITEN employs a hierarchical control scheme with a high-level VLM (GPT-5-mini) for reasoning and assessment, and a pre-trained low-level VLA (π^0.5^-DROID) for subtask execution.",
          "The method uses an iterative, two-phase loop: a Reasoning Phase for plan generation and execution, and an Assessment Phase for evaluating outcomes and generating structured feedback.",
          "A VLM 'judge' uses a chain of prompts to diagnose subtask success/failure, describe actual outcomes, and reason about failure causes, storing this structured feedback as 'in-context experiences' for subsequent planning."
        ],
        "keyInsights": [
          "Learning from failures, not just successes, is critical for efficient and robust adaptation in real-world robot learning.",
          "Structured assessment procedures, guiding the VLM judge with specific inputs (e.g., first and last observation images) and questions, are essential for accurate diagnosis of physical outcomes, as VLMs struggle with raw, unstructured video interpretation.",
          "The approach allows high-level VLMs to learn the specific affordances and biases of a given robotic embodiment and low-level policy through interaction, grounding abstract knowledge in physical reality at inference time."
        ],
        "results": [
          "LITEN consistently increased success rates over five iterative attempts across three complex manipulation tasks (Stacking, Emptying Bowls, Moving Off Table).",
          "It significantly outperformed baselines, including a 'No-Feedback' approach, a 'Positive-ICL' method (only learning from successes), and a direct 'Reflexion' adaptation which failed due to poor raw video comprehension.",
          "Ablation studies confirmed the critical importance of the detailed 'failure reasoning' and 'outcome analysis' steps within the structured assessment phase for effective learning and adaptation."
        ]
      },
      "image_url": "image/2510.19752v1.png",
      "universal_paper_id": "2510.19752",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 48,
          "last_7_days": 48
        },
        "public_total_votes": 8
      },
      "first_publication_date": "2025-10-22T16:43:29.000Z",
      "publication_date": "2025-10-22T16:43:29.000Z",
      "updated_at": "2025-10-23T02:08:58.724Z",
      "topics": [
        "agents",
        "chain-of-thought",
        "Computer Science",
        "cs.AI",
        "cs.RO",
        "multi-modal-learning",
        "reasoning",
        "reinforcement-learning",
        "robotic-control",
        "test-time-inference",
        "vision-language-models"
      ],
      "organization_info": [
        {
          "name": "UC Berkeley",
          "image": "images/organizations/berkeley.png"
        },
        {
          "name": "Physical Intelligence",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 4,
      "github_url": "https://github.com/ameesh-shah/liten-vla",
      "distance": 1
    },
    {
      "id": "019a13de-aaa3-7fd6-b7b7-bc8b6041c3f3",
      "paper_group_id": "019a13de-aaa3-7fd6-b7b7-bc8b6041c3f3",
      "title": "Every Question Has Its Own Value: Reinforcement Learning with Explicit Human Values",
      "abstract": "我们提出了显式人类价值强化学习（RLEV）的方法，该方法将大型语言模型（LLM）的优化与可量化的人类价值信号直接对齐。虽然可验证奖励的强化学习（RLVR）在使用二元正确性奖励的客观领域中有效地训练模型，但它忽视了并非所有任务都是同等重要的。RLEV通过将人类定义的价值信号直接纳入奖励函数，扩展了这一框架。使用具有显式真实值标签的考试风格数据，RLEV在多个强化学习算法和模型规模中始终优于仅以正确性为基础的基准。至关重要的是，RLEV 策略不仅提高了价值加权的准确性，而且还学习了一种对价值敏感的终止策略：对于低价值的提示简洁，对于高价值的提示则全面。我们证明这种行为源于对序列结束标记的价值加权梯度放大。消融研究确认了这种增益与价值对齐之间存在因果关系。RLEV在嘈杂的价值信号下仍然表现稳健，例如基于难度的标签，表明优化显式效用函数为将LLM与人类优先事项对齐提供了切实可行的路径。",
      "paper_summary": {
        "summary": "Reinforcement Learning with Explicit Human Values (RLEV), developed by Tencent AI Lab, integrates quantifiable human value signals directly into the reward function for Large Language Model (LLM) alignment. This enables LLMs to optimize for total human-defined utility rather than just correctness, leading to higher value-weighted accuracy and strategic response generation that prioritizes high-value tasks.",
        "originalProblem": [
          "Existing Reinforcement Learning with Verifiable Rewards (RLVR) methods treat all correct answers as equally valuable, failing to account for varying human-defined importance across tasks.",
          "LLMs trained solely on correctness count are not optimized for the total score or overall human utility in real-world applications where tasks have non-uniform importance.",
          "Dominant alignment methods like RLHF are computationally expensive and may be inefficient for tasks with objectively verifiable ground truth."
        ],
        "solution": [
          "RLEV proposes a human utility function, U(x, y) = v(x) \n 1_{correct}(y), where v(x) is the intrinsic human-defined value of prompt x, and 1_{correct}(y) indicates correctness.",
          "A practical surrogate reward function, r(x, y) = s(x) \n 1_{correct}(y), is designed where s(x) is an additive and clipped scaling factor (1 + min(alpha \n v(x), 1)) derived from normalized human values to ensure stable training.",
          "The method fine-tunes LLMs using standard RL algorithms (REINFORCE++, RLOO, GRPO), maximizing the expected value-weighted reward to align with explicit human priorities, primarily on an exam-style question-answering dataset."
        ],
        "keyInsights": [
          "The human-aligned scaling factor directly amplifies the policy gradient magnitude, particularly for End-of-Sequence (EOS) tokens, leading to a value-sensitive termination policy in LLMs.",
          "LLMs learn to strategically allocate their 'token budget', generating concise responses for low-value prompts and thorough ones for high-value prompts, demonstrating 'judicious' behavior.",
          "The chosen additive and clipped reward function, coupled with value normalization, provides stability and effectiveness during training, especially with highly skewed distributions of human values."
        ],
        "results": [
          "RLEV consistently achieved higher Human-Aligned Accuracy (H-Acc) than correctness-only baselines, with average gains of 2.0% for 7B models and 2.8% for 32B models.",
          "RLEV models demonstrated a dramatic reduction in average response length (e.g., from 246.9 to 98.6 tokens for 32B models) while simultaneously increasing Value Density due to learned value-sensitive termination.",
          "The method proved robust even with noisy or approximate human value signals (e.g., difficulty-based weak labels or predicted values), consistently outperforming correctness-only baselines and indicating broad practical applicability."
        ]
      },
      "image_url": "image/2510.20187v1.png",
      "universal_paper_id": "2510.20187",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 42,
          "last_7_days": 42
        },
        "public_total_votes": 11
      },
      "first_publication_date": "2025-10-23T04:15:22.000Z",
      "publication_date": "2025-10-23T04:15:22.000Z",
      "updated_at": "2025-10-24T01:39:11.139Z",
      "topics": [
        "agents",
        "Computer Science",
        "cs.CL",
        "cs.LG",
        "fine-tuning",
        "human-ai-interaction",
        "optimization-methods",
        "reasoning",
        "reinforcement-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "Tencent AI Lab",
          "image": null
        },
        {
          "name": "Princeton University",
          "image": "images/organizations/princeton.jpg"
        }
      ],
      "author_info": [],
      "github_stars": null,
      "github_url": null,
      "distance": 1
    },
    {
      "id": "019a158c-0b90-77ab-a1c0-5edbf629672b",
      "paper_group_id": "019a158c-0b90-77ab-a1c0-5edbf629672b",
      "title": "DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion",
      "abstract": "扩散变换器模型可以生成具有卓越保真度和细节的图像，但由于自注意机制在图像标记数目上的二次扩展，训练它们在超高分辨率下仍然成本极高。本文介绍了一种动态位置外推（DyPE）的新方法，这是一种无训练的方式，能使预训练的扩散变换器在远超其训练数据的分辨率下合成图像，而无需额外的采样成本。DyPE利用了扩散过程固有的频谱进程，其中低频结构早期收敛，而高频结构需要更多步骤才能解析。具体来说，DyPE在每次扩散步骤中动态调整模型的位置信息编码，使其频谱与生成过程的当前阶段匹配。这种方法使我们能够在超过训练分辨率的情况下生成图像，例如，使用FLUX生成1600万像素的图像。在多个基准测试中，DyPE始终提升性能，并在超高分辨率图像生成中实现了最先进的保真度，且在更高分辨率下提升效果更为显著。项目页面可以在此https URL查看。",
      "paper_summary": {
        "summary": "Researchers at The Hebrew University of Jerusalem introduced DYPE, a training-free method that enables pre-trained Diffusion Transformers (DiTs) to generate images at ultra-high resolutions without additional training or inference overhead. It achieves this by dynamically adjusting positional encoding based on the spectral evolution of the diffusion process, leading to improved image fidelity and detail compared to static extrapolation techniques.",
        "originalProblem": [
          "Training Diffusion Transformer (DiT) models directly at ultra-high resolutions (e.g., 4096x4096 and beyond) is prohibitively expensive due to the quadratic complexity of self-attention.",
          "Existing static positional encoding (PE) extrapolation methods in transformers (e.g., PI, NTK-aware, YaRN) do not account for the dynamic spectral progression inherent to the diffusion process, limiting their ability to maintain quality at extreme resolution increases.",
          "The inability to efficiently scale pre-trained DiTs to very high resolutions restricts their application in scenarios requiring highly detailed visual outputs."
        ],
        "solution": [
          "DYPE leverages a Fourier-space analysis of the reverse diffusion process, revealing that low-frequency components converge early, while high-frequency details evolve throughout the denoising steps.",
          "It introduces a time-dependent scaling function, `κ(t)`, which dynamically modulates existing positional extrapolation formulas (e.g., PI, NTK-aware, YaRN).",
          "This dynamic adjustment ensures that positional encoding emphasizes broader context (via stronger scaling) early in the diffusion process and then shifts to better represent high-frequency details (via reduced scaling) as denoising progresses."
        ],
        "keyInsights": [
          "The spectral content of images generated by diffusion models evolves significantly over time, with low frequencies converging much earlier than high frequencies.",
          "Effective resolution extrapolation for Diffusion Transformers requires a dynamic positional encoding strategy that adapts to the instantaneous spectral needs of the denoising process, rather than a fixed, static approach.",
          "By aligning the positional encoding's frequency emphasis with the current stage of diffusion, it is possible to maintain both structural coherence and fine-grained detail at resolutions far beyond the training data."
        ],
        "results": [
          "DYPE consistently and significantly improved ultra-high-resolution text-to-image generation on FLUX and class-to-image generation on FiTv2, outperforming static baselines across metrics like CLIPScore, ImageReward, Aesthetics, and FID at resolutions up to 4096x4096 and beyond.",
          "Human evaluations showed strong preference for DYPE's outputs (DY-NTK-aware and DY-YaRN) over their static counterparts in terms of text alignment, structural coherence, and fine details at 4096x4096 resolution.",
          "The method demonstrated enhanced stability and artifact mitigation at increasing resolutions, with DYPE (DY-YaRN) maintaining quality up to 6144x6144 while baselines degraded sharply at lower resolutions, and proved effective for panoramic image generation."
        ]
      },
      "image_url": "image/2510.20766v1.png",
      "universal_paper_id": "2510.20766",
      "metrics": {
        "total_votes": 2,
        "visits_count": {
          "all": 28,
          "last_7_days": 28
        },
        "public_total_votes": 7
      },
      "first_publication_date": "2025-10-23T17:42:14.000Z",
      "publication_date": "2025-10-23T17:42:14.000Z",
      "updated_at": "2025-10-24T09:28:10.896Z",
      "topics": [
        "Computer Science",
        "cs.CV",
        "efficient-transformers",
        "generative-models",
        "image-generation",
        "inference-optimization",
        "transfer-learning",
        "transformers"
      ],
      "organization_info": [
        {
          "name": "The Hebrew University of Jerusalem",
          "image": null
        }
      ],
      "author_info": [],
      "github_stars": 47,
      "github_url": "https://github.com/guyyariv/DyPE",
      "distance": 1
    },
    {
      "id": "019a17b9-b4cb-7f1d-b3a9-061835976a87",
      "paper_group_id": "019a17b9-b4cb-7f1d-b3a9-061835976a87",
      "title": "Smooth sets of fields: A pedagogical introduction",
      "abstract": "为了为描述物理理论中出现的许多不同场域提供良好的范畴设置，本文对光滑集的范畴概念进行了教学性介绍，并讨论了光滑集上的拓扑的一些简单性质。通过切向函子和变分双复形这两个具体例子，说明了将几何结构引入这些空间的过程。",
      "paper_summary": null,
      "image_url": "image/2510.20422v1.png",
      "universal_paper_id": "2510.20422",
      "metrics": {
        "total_votes": 0,
        "visits_count": {
          "all": 14,
          "last_7_days": 14
        },
        "public_total_votes": 3
      },
      "first_publication_date": "2025-10-23T10:49:39.000Z",
      "publication_date": "2025-10-23T10:49:39.000Z",
      "updated_at": "2025-10-24T19:37:17.771Z",
      "topics": [
        "math-ph",
        "Physics"
      ],
      "organization_info": [
        {
          "name": "Universidad Carlos III de Madrid",
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