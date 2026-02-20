Abstract
The traffic assignment problem is essential for traffic flow analysis, traditionally solved using mathematical programs under the Equilibrium principle. These methods become computationally prohibitive for large-scale networks
due to non-linear growth in complexity with the number of OD pairs. This study introduces a novel data-driven approach using deep neural networks, specifically leveraging the Transformer architecture, to predict equilibrium path
flows directly. By focusing on path-level traffic distribution, the proposed model captures intricate correlations between OD pairs, offering a more detailed and flexible analysis compared to traditional link-level approaches. The
Transformer-based model drastically reduces computation time, while adapting to changes in demand and network structure without the need for recalculation. Numerical experiments are conducted on the Manhattan-like synthetic
network, the Sioux Falls network, and the Eastern-Massachusetts network. The results demonstrate that the proposed model is orders of magnitude faster than conventional optimization. It efficiently estimates path-level traffic flows
in multi-class networks, reducing computational costs and improving prediction accuracy by capturing detailed trip and flow information. The model also adapts flexibly to varying demand and network conditions, supporting traffic
management and enabling rapid ‘what-if’ analyses for enhanced transportation planning and policy-making.
Keywords: Traffic Assignment, Transformer, Network equilibrium, Traffic flow prediction, Path-based analysis

To read in detail about the work, refer https://arxiv.org/pdf/2510.19889.

@misc{ameli2025optimizationpredictiontransformerbasedpathflow,
      title={From Optimization to Prediction: Transformer-Based Path-Flow Estimation to the Traffic Assignment Problem}, 
      author={Mostafa Ameli and Van Anh Le and Sulthana Shams and Alexander Skabardonis},
      year={2025},
      eprint={2510.19889},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.19889}, 
}
