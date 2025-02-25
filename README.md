
---

# DSPy: Programming—Not Prompting—Language Models

![](https://github.com/stanfordnlp/dspy/raw/main/docs/docs/static/img/dspy_logo.png)

DSPy is a powerful framework designed to shift the paradigm from prompting language models (LMs) with brittle, hard-to-maintain strings to **programming structured and modular AI systems**. It enables developers to build reliable and scalable AI applications while optimizing prompts and model weights dynamically.

## 🚀 Why DSPy?

### ✅ **Modular AI System Development**
DSPy replaces traditional string-based prompting with structured modules, allowing you to build compositional and optimizable AI pipelines.

### 🔄 **Fast Iteration & Optimization**
With DSPy, you can quickly iterate on AI behavior without manually tweaking prompts. It supports **self-improving** models through integrated optimizers.

### 🤖 **Supports Multiple LMs**
Works seamlessly with **OpenAI, Anthropic, Databricks**, and local models (running on CPU/GPU servers) with simple API configurations.

### 🏗 **From Simple Scripts to Advanced AI Systems**
DSPy scales from small proof-of-concept scripts to full-fledged **Retrieval-Augmented Generation (RAG)**, **Classification**, and **Agent-based** applications.

---
## 🛠 Getting Started

### 🔹 Install DSPy

```bash
pip install -U dspy
```

### 🔹 Set Up Your Language Model (LM)

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
dspy.configure(lm=lm)
```

DSPy supports various LMs, including OpenAI, Anthropic, and local LMs. You can authenticate by setting the `OPENAI_API_KEY` environment variable or passing the API key explicitly.

---

## 📖 Building DSPy Modules

DSPy allows you to define **AI components as structured modules**, rather than dealing with unstructured prompt strings. Below are examples of how to build **math solvers, RAG systems, classifiers, and agents** using DSPy.

### 🧮 **Math Reasoning with DSPy**
```python
import dspy
math = dspy.ChainOfThought("question -> answer: float")

result = math(question="Two dice are tossed. What is the probability that the sum equals two?")
print(result)
```
#### ✅ Possible Output:
```
Prediction(
    reasoning='When two dice are tossed, each die has 6 faces, resulting in 6 x 6 = 36 possible outcomes...'
    answer=0.0277776
)
```

---
### 🔍 **Retrieval-Augmented Generation (RAG)**
```python
import dspy
from dspy.datasets import HotPotQA

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ReAct("question -> answer", tools=[search_wikipedia])
print(rag("What is DSPy?"))
```

---
### 📊 **Classification with DSPy**
```python
class SentimentClassifier(dspy.Module):
    def forward(self, text):
        return dspy.Predict("text -> sentiment: {'positive', 'neutral', 'negative'}")(text)

classifier = SentimentClassifier()
print(classifier.forward("I love using DSPy!"))
```

---
## 🎯 Optimizing DSPy Programs

### 🔹 **Self-Improving AI with DSPy Optimizers**
DSPy optimizes AI pipelines with built-in **optimizers** that refine prompts, improve outputs, and even fine-tune weights.

#### 🔥 **Example: Optimizing a ReAct Agent**
```python
import dspy
from dspy.datasets import HotPotQA

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]

react = dspy.ReAct("question -> answer")
tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
optimized_react = tp.compile(react, trainset=trainset)
```
**Result:** Optimized ReAct performance increased from **24% to 51% accuracy** using **DSPy MIPROv2** optimizer.

---

## 🏆 DSPy in Research & Open-Source AI

DSPy emerged from **Stanford NLP in 2022**, evolving from research in **modular LM architectures, inference strategies, and self-optimizing AI**. Today, the DSPy community actively contributes to innovations in:

✅ **Optimizers**: MIPROv2, BetterTogether, LeReT  
✅ **Architectures**: STORM, IReRa, DSPy Assertions  
✅ **Applications**: PAPILLON, PATH, WangLab@MEDIQA, UMD's Prompting Study  

DSPy empowers **open-source AI development**, allowing modular and compositional approaches for improving **LLM-based AI systems** over time.

---
## 📬 Get Involved

🚀 **Contribute**: Join our GitHub repo and submit PRs!  
📢 **Community**: Engage in discussions on our Discord server.  
📄 **Documentation**: Read the latest DSPy guides.  

---
## 📌 Contact

📧 **Email**: saadsalmanakram1@gmail.com  
🌐 **GitHub**: [SaadSalmanAkram](https://github.com/saadsalmanakram)  
💼 **LinkedIn**: [Saad Salman Akram](https://www.linkedin.com/in/saadsalmanakram/)  

---

**🔥 Get Started with DSPy Today! 🔥**

