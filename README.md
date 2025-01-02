<div align="center">

# Process Reinforcement Through Implicit Rewards

<p align="center">
    <a href="#links"> Links</a> •
    <a href="#introduction"> Introduction</a> •
    <a href="#evaluation">Evaluation</a>
</p>


</div>

# Links

- 📜 [Blog]()
- 🤗 [PRIME Collection](https://huggingface.co/PRIME-RL)

# Introduction

Advanced reasoning of large language models (LLMs), while improvable through data-driven imitation, is still clouded by serious scalability challenges. We believe the key to overcoming such challenges lies in transforming data-driven approaches into *exploration-based* methods, as exemplified by reinforcement learning (RL). To this end, two critical bottlenecks need to be alleviated to bridge this transformation: (1) how to obtain precise *reward signals* efficiently and scalably, especially for *dense* ones? (2) how can we build effective RL algorithms to fully *unleash* the potential of these signals? 


We seek the **scalable** path towards advanced reasoning capabilities with **efficient reward modeling and reinforcement learning**. Our work stems from the implicit process reward modeling (PRM) objective. Without the need for any process label, implicit PRM is trained as an outcome reward model (ORM) and then used as a PRM. Besides improving model performance through inference scaling, the true power of the implicit PRM is unveiled in online RL training. Specifically, it brings three benefits to RL:
- **Dense Reward:** Implicit PRM directly learns a Q-function that provides rewards for *each token*, which alleviates the reward sparsity issue without the need of an extra value model.
- **Scalability:** Implicit PRM can be online updated with only outcome label. Therefore, we can directly update the PRM with on-policy rollouts given outcome verifiers, which mitigates the distribution shift as well as scalability issues for PRMs.
- **Simplicity:** Implicit PRM is inherently a language model. In practice, we show that it is unnecessary to train a PRM beforehand, since the SFT model itself already serves as a strong starting point.

We then dive into RL to figure out its key algorithm designs and implementation techniques. To this end, we present Process Reinforcement through IMplicit rEwards, PRIME, which effectively incorporates and updates PRMs in RL. 


<img src="./figures/prm.gif" alt="prm" style="zoom: 33%;" />

As shown in the animation above, in PRIME, the policy model and PRM are both initialized with the SFT model. For each RL iteration, the policy model first generates rollouts. Then, the [implicit PRM](https://arxiv.org/abs/2412.01981) and outcome verifier score the rollouts, and the implicit PRM gets updated on the rollouts with the outcome reward. Finally, the outcome reward \\(r_o\\) and process reward \\(r_p\\) are combined and used to update the policy model. 

The PRIME implementation pseudocode is as follows:

<img src="./figures/prime-algo.png" alt="prime-algo" style="zoom: 50%;" />

The algorithm flow includes:

1. **Prompt filtering** based on policy model performance, only preserving those on which the policy model \\(\pi_\theta\\) achieves a accuracy between 0.2 and 0.8.
2. **Calculate implicit process reward** \\(r^t\\).
3. **Update Implicit PRM** \\(\pi_\psi\\) based on predicted implicit process reward \\(r^t\\) and ground truth outcome label \\(r\\).
4. **Advantage estimation with RLOO.** Specifically, we first calculate the return of outcome rewards and implicit process rewards separately:

- For ground truth outcome rewards, we directly adopt RLOO without any modification.

- For implicit process rewards, we perform a three-step process to calculate return: (1) Use the averaged implicit process rewards to calculate the leave-one-out baseline (2) Normalize the process reward at step \\(t\\) by subtracting the baseline; (3) Calculate the discounted return for each response.

  Finally, advantage is set to the combination of both returns. 

​    5. **Update the policy** \\(\pi_\theta\\) using PPO loss for legit importance sampling.


# Usage
We apply tailored prompts for coding and math task:
**System Prompt**
```
\nWhen tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process.\n\n[ASSESS]\n\n[ADVANCE]\n\n[VERIFY]\n\n[SIMPLIFY]\n\n[SYNTHESIZE]\n\n[PIVOT]\n\n[OUTPUT]\n\nYou should strictly follow the format below:\n\n[ACTION NAME]\n\n# Your action step 1\n\n# Your action step 2\n\n# Your action step 3\n\n...\n\nNext action: [NEXT ACTION NAME]\n
```
**Coding**
```
{question} + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.
```
**Math**
```
{question} + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
```
# Evaluation
Through PRIME, we successfully achieve substantial improvements on key reasoning benchmarks over our SFT version of the model, leading to **16.7%** improvement on average, and over **20%** on AMC&AIME competitions. Our final model Eurus-2-7B-PRIME, based on Qwen-2.5-Math-7B-Base, surpassed its instruct version on 5 key reasoning benchmarks. 
The final results are presented below:
|               | **Eurus-2-7B-PRIME** | **Eurus-2-7B-SFT** | **Qwen-2.5-Math-7B-Instruct** | **Llama-3.1-70B-Instruct** | **GPT-4o** |
| ------------- | -------------------- | ------------------ | ----------------------------- | -------------------------- | ---------- |
| AIME 2024     | **26.7 (+23.3)**     | 3.3                | 13.3                          | 16.7                       | 9.3        |
| MATH-500      | 79.2 (+14.1)         | 65.1               | **79.8**                      | 64.6                       | 76.4       |
| AMC           | **57.8 (+27.7)**     | 30.1               | 50.6                          | 30.1                       | 45.8       |
| Minerva Math  | **38.6 (+5.9)**      | 32.7               | 34.6                          | 35.3                       | 36.8       |
| OlympiadBench | 42.1 (+12.3)         | 29.8               | 40.7                          | 31.9                       | **43.3**   |
| Avg.          | **48.9 (+ 16.7)**    | 32.2               | 43.8                          | 36.4                       | 43.3       |

![image-20241230162026156](./figures/performance.jpg)

We achieved this with only 1/10 data and model resources compared with Qwen-Math.
|            | **Eurus-2-7B-PRIME**               | **Qwen2.5-Math-7B-Instruct**    |
| ---------- | ---------------------------------- | ------------------------------- |
| Base Model | Qwen2.5-Math-7B                    | Qwen2.5-Math-7B                 |
| SFT Data   | **230K (open-source)**             | 2.5M (open-source and in-house) |
| RM Data    | **0**                              | 618K (in-house)                 |
| RM         | **Eurus-2-7B-SFT**                 | Qwen2.5-Math-RM (72B)           |
| RL Data    | **150K queries  \\(\times\\)4 samples** | 66K queries \\(\times\\) 32 samples |

# Citation
If you find PRIME or ImplicitPRM helpful, please cite us.

```
@misc{cui2024process,
  title={Process Reinforcement through Implicit Rewards},
  author={Ganqu Cui, Lifan Yuan, Zefan Wang, Wendi Li, Hanbin Wang, Tianyu Yu, Bingxiang He, Yuchen Fan, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, Ning Ding},
  year={2025},
  howpublished={\url{}},
  note={Notion Blog}
}
```

```
@article{yuan2024implicitprm,
  title={Free Process Rewards without Process Labels},
  author={Lifan Yuan and Wendi Li and Huayu Chen and Ganqu Cui and Ning Ding and Kaiyan Zhang and Bowen Zhou and Zhiyuan Liu and Hao Peng},
  journal={arXiv preprint arXiv:2412.01981},
  year={2024}
}
```