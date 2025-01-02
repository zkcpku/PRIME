## RL Data Preprocessing

We curated a high-quality RL training dataset of mathematics and coding problems with outcome verifiers (LaTeX answers for math and test cases for coding).

- For math, we sourced from [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT),  which contains about 860K math problems. The problems span from Chinese high school mathematics to International Mathematical Olympiad competition questions.
- For coding, we sourced from [APPS](https://huggingface.co/datasets/codeparrot/apps), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), [TACO](https://huggingface.co/datasets/BAAI/TACO), and [Codeforces](https://huggingface.co/datasets/MatrixStudio/Codeforces-Python-Submissions).

To further increase data quality, we conducted detailed cleaning and filtering. Finally, we retain 457k math problems and 27k coding problems. Here we present our four-stage RL math data preprocessing process.

### Stage 1: Data Filtering and Question-Type Classification

The preprocessing pipeline employs a systematic rule-based approach to filter and classify mathematical problems to create a high-quality dataset with solvable problems, appropriate difficulty levels, and correct solutions.

We exclude problems containing figures or diagrams since they require visual processing capabilities. We also remove proof questions due to difficulties in answer verification. The remaining problems are classified into question-answering, multiple-choice, or fill-in-the-blank questions based on specific patterns. Since fill-in-the-blank questions comprise less than 400 examples compared to the much larger set of multiple-choice questions, we focus solely on multiple-choice questions for further processing.

To use the code, first download [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), set the dataset path in `stage1_filter.py` and then run:

```shell
python stage1_filter.py
```

### Stage 2: Converting to Direct Question-Answer Format

We transform multiple-choice questions into a direct question-answer format through three sequential stages: rule-based filtering, LLM-based filtering, and LLM-based formatting.

We first identify and remove questions that inherently require multiple-choice options - specifically, those where comparing specific statements or properties is essential to the problem-solving process. These questions cannot be meaningfully converted to a direct question-answer format. The initial filtering employs simple rule-based pattern matching, searching for keywords like "following" and "statement" that typically indicate option-dependent problems.

Following the rule-based filtering, we employ Llama-3.1-8B-Instruct to perform a more nuanced classification of the remaining questions. Our pilot study revealed that while the LLM occasionally misclassifies questions, it tends to err on the conservative side - marking potentially convertible questions as requiring options rather than the reverse. Given our large dataset, we accepted this conservative approach to maintain quality.

For questions classified as convertible, we implement a two-phase reformatting process:

1. Question Reformatting: Removing choice indicators and restructuring the question to elicit direct answers
2. Solution Reformatting: Converting multiple-choice solutions into step-by-step derivations, ensuring all final answers are presented in standard LaTeX boxed format

This systematic approach maintains mathematical rigor while creating a standardized format suitable for downstream applications.

To use the code, first correctly set the data path in `stage2_format_choice.py` and then run:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python stage2_format_choice.py
```

### Stage 3: Merge

we simply merge all question-answer pairs and conduct a data filtering again.

To use the code, first correctly set the data path in `stage3_merge.py` and then run:

```shell
python stage3_merge.py
```

### Stage 4: Problem and Solution Validation

The final stage involves merging all question-answer pairs and performing LLM-based comprehensive validation. We identify two key aspects in validation: solvability and correctness.

We leverage state-of-the-art mathematical reasoning models, including QwQ-32B-Preview and Qwen2.5-Math-72B-Instruct, employing a self-consistency approach to determine problem solvability, and if solvable, verify the correctness of solutions provided in the original dataset.

To enhance validation accuracy, we first analyzed sample problems to identify characteristics of solvable and unsolvable cases and created synthetic unsolvable problems featuring missing conditions or logical contradictions. Based on these samples, we developed specialized prompts to improve the models' ability to distinguish solvability.

Each problem undergoes five independent validation attempts, where the LLM:

1. Provides step-by-step solutions using LaTeX formatting
2. Identifies insolvability due to missing conditions or logical contradictions
3. Generates complete reasoning traces for solvable problems
4. Presents final answers in standardized LaTeX boxed format (`\\boxed{}`)
5. Documents any impediments to solution completion 

We evaluate two key consistency measures across multiple validation attempts:

- Status Consistency: Agreement on problem solvability
- Answer Consistency:
  - Consistency of solutions across different attempts
  - Agreement between generated solutions and ground truth

The final dataset retains only problems that demonstrate:

- Consistent solvability across validation attempts
- Agreement in solutions across multiple attempts
- Alignment with ground truth answers

This rigorous validation process ensures the resulting dataset comprises well-defined, solvable problems with verified, accurate solutions.

To use the code, first correctly set the data path in `stage4_judge.py` and then run:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python stage4_judge.py
```

You can use different models to judge the correctness by specifying the `model_path` parameter, e.g. Qwen2.5-Math-72B-Instruct and QwQ-32B-Preview.