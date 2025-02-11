# Reflect, Reason, Rephrase ($\textup{R}^3$-Detox) : An In-Context Learning Approach to Text Detoxification
![My SVG Image](https://raw.githubusercontent.com/your-username/your-repo/main/path-to-image.svg)
## Abstract
Traditional content moderation censors harmful content, but often limits user participation. Text detoxification offers a better alternative, promoting civility without silencing voices. However, prior approaches oversimplify the task by treating detoxification as a one-step process, neglecting the deep contextual analysis needed to remove toxicity while preserving meaning. In this paper, we introduce $\textup{R}^3$-Detox—a Reflect, Reason, and Rephrase framework that enhances detoxification through a structured three-step process, all executed within a single prompt. First, we instruct the LLM to analyze potential toxic words or phrases, guided by Shapley values from toxicity detectors, to counteract potential hallucinations. Next, the model assesses the overall toxicity of the sentence based on these identified elements. Finally, leveraging this prior analysis, the model reasons about necessary modifications to eliminate toxicity while maintaining meaning. We apply this framework and Self-Reflection models to enrich offensive content paraphrasing datasets—ParaDetox, Parallel Detoxification, and APPDIA—by adding explicit detoxification reasoning to each instance, which originally contained only input sentences and their paraphrases. We evaluate our methodology using In-Context Learning, comparing $\textup{R}^3$-Detox against state-of-the-art methods on the same datasets. Experimental results show that our approach outperforms existing methodologies, even in instruction-following models.
## To-Do List ✅

- [ ] Upload the Data 📂
- [ ] Upload the Code 📝

## Data
