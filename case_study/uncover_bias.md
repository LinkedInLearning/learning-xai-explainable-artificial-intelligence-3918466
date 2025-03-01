# Responsible AI Testing Framework

## Background: Addressing Bias in Generative AI

As highlighted in the University of Michigan study on CLIP (Contrastive Language-Image Pre-training), generative AI models often exhibit systematic biases in representation:

- Western cities appear modern and developed, while non-Western locations are disproportionately associated with underdevelopment
- Models associate wealth and technological advancement with Western countries while depicting developing regions with poverty
- Occupational bias exists, with Western-based jobs given more prominence
- Specific cultural norms and aesthetics are overrepresented, sidelining diverse global perspectives
- Training data originates largely from wealthier, English-speaking countries, creating representation imbalances

These biases directly impact how people and places from different regions are represented in AI-generated content across journalism, education, and media. As AI becomes increasingly influential in shaping narratives and decision-making, responsible AI development is essential.

## Enhancing Your Analysis with Modern LLMs

The Jupyter notebooks provided in this course demonstrate various techniques for evaluating and mitigating bias using NLTK. This guide will help you enhance your analysis by incorporating more powerful language models:

1. Microsoft Phi-4 (free, available through Hugging Face)
2. OpenAI models (requires API key/subscription)
3. Anthropic Claude (requires API key/subscription)

## Getting Started with [Microsoft Phi-4](https://huggingface.co/microsoft/phi-4)

Microsoft Phi-4 is a powerful yet efficient language model that's freely available through Hugging Face. It provides an excellent option for text analysis when you don't want to incur costs.

### Installation

```bash
pip install transformers torch huggingface_hub
```

### Basic Usage with Your Existing Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_with_phi4(texts, prompt_template=None):
    """
    Analyze texts using Microsoft Phi-4
    
    Args:
        texts: List of text strings to analyze
        prompt_template: Optional template for structuring the analysis request
        
    Returns:
        List of model responses
    """
    # Load model and tokenizer
    model_name = "microsoft/phi-4"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype="auto"
    )
    
    results = []
    
    for text in texts:
        # Create prompt
        if prompt_template:
            prompt = prompt_template.format(text=text)
        else:
            prompt = f"Analyze the sentiment and potential bias in this text: {text}"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(response)
    
    return results
```

### Integration Example for Counterfactual Testing

Here's how you could integrate Phi-4 with the counterfactual testing approach from the notebooks:

```python
# Sample code - adapt to your specific analysis
original_text = "The CEO gave a presentation to the board."
variations = [
    "The female CEO gave a presentation to the board.",
    "The African CEO gave a presentation to the board.",
    "The young CEO gave a presentation to the board."
]

# Analyze original and variations
original_analysis = analyze_with_phi4([original_text])[0]
variation_analyses = analyze_with_phi4(variations)

# Compare responses to identify potential biases
for i, variation in enumerate(variations):
    print(f"Variation: {variation}")
    print(f"Analysis: {variation_analyses[i][:100]}...")  # Truncated output
    print("---")
```

## Using Commercial LLMs: OpenAI and Claude

For more powerful analysis capabilities, you can use commercial LLMs. Note that these require API keys and will incur usage costs.

### [OpenAI](https://platform.openai.com/docs/overview) Integration

```python
import openai

def analyze_with_openai(texts, model="gpt-4o", system_prompt=None):
    """
    Analyze texts using OpenAI models
    
    Args:
        texts: List of text strings to analyze
        model: OpenAI model to use
        system_prompt: Optional system prompt to guide analysis
        
    Returns:
        List of model responses
    """
    # Set your API key
    openai.api_key = "your-api-key"  # Replace with your actual key
    
    if system_prompt is None:
        system_prompt = """
        Analyze the following text for potential biases related to gender, ethnicity, 
        age, socioeconomic status, or regional representation. Identify any stereotypes 
        or imbalanced portrayals.
        """
    
    results = []
    
    for text in texts:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        results.append(response.choices[0].message.content)
    
    return results
```

### Anthropic [Claude](https://docs.anthropic.com/en/release-notes/api) Integration

```python
from anthropic import Anthropic

def analyze_with_claude(texts, model="claude-3-5-sonnet", system_prompt=None):
    """
    Analyze texts using Anthropic Claude
    
    Args:
        texts: List of text strings to analyze
        model: Claude model to use
        system_prompt: Optional system prompt to guide analysis
        
    Returns:
        List of model responses
    """
    # Initialize client
    anthropic = Anthropic(api_key="your-api-key")  # Replace with your actual key
    
    if system_prompt is None:
        system_prompt = """
        Analyze the following text for potential biases related to gender, ethnicity, 
        age, socioeconomic status, or regional representation.
        """
    
    results = []
    
    for text in texts:
        message = anthropic.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=500,
            temperature=0.3,
            messages=[
                {"role": "user", "content": text}
            ]
        )
        
        results.append(message.content[0].text)
    
    return results
```

## Practical Applications for Your Existing Code

### 1. Enhanced Counterfactual Testing

Implement counterfactual testing by systematically altering demographic attributes in your prompts and analyzing how model outputs change.

```python
# Define demographic variations to test
demographic_variations = {
    "gender": ["male", "female", "non-binary"],
    "ethnicity": ["Asian", "Black", "Hispanic", "White"],
    "age": ["young", "middle-aged", "elderly"],
    "region": ["Western", "Eastern", "African", "South American"]
}

# Template for creating counterfactual examples
template = "A {age} {gender} {ethnicity} professional from {region} applied for the job."

# Generate all counterfactual examples
import itertools
counterfactuals = []
attributes = []

for values in itertools.product(*demographic_variations.values()):
    attr_dict = dict(zip(demographic_variations.keys(), values))
    counterfactuals.append(template.format(**attr_dict))
    attributes.append(attr_dict)

# Analyze with your chosen model (Phi-4, OpenAI, or Claude)
results = analyze_with_phi4(counterfactuals)  # Or any other analysis function

# Compare results across demographic variations
# You can implement metrics to quantify differences in sentiment, complexity, etc.
```

### 2. Stratified Sampling for Model Evaluation

Ensure your test data includes diverse, well-represented perspectives across demographic dimensions:

```python
from sklearn.model_selection import StratifiedKFold

# Assuming df is your dataset with demographic columns
demographic_cols = ['gender', 'age_group', 'region', 'language']

# Create stratification variable
df['strata'] = df[demographic_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)

# Perform stratified sampling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(df, df['strata']):
    train_set = df.iloc[train_idx]
    test_set = df.iloc[test_idx]
    
    # Verify distribution
    for col in demographic_cols:
        print(f"Training set {col} distribution:", 
              train_set[col].value_counts(normalize=True))
        print(f"Test set {col} distribution:", 
              test_set[col].value_counts(normalize=True))
```

### 3. Model Constraints and Bias Mitigation

Implement prompt engineering techniques to reduce biased outputs:

```python
# Example of fairness-aware prompt engineering
def create_fairness_aware_prompt(base_prompt):
    """Add fairness constraints to a prompt"""
    fairness_constraints = [
        "Ensure your response treats all demographic groups equally.",
        "Avoid stereotypes related to gender, ethnicity, age, or region.",
        "Use inclusive language that respects diverse perspectives.",
        "Consider how your response might impact underrepresented groups."
    ]
    
    constraint_text = " ".join(fairness_constraints)
    enhanced_prompt = f"{base_prompt}\n\nImportant: {constraint_text}"
    
    return enhanced_prompt

# Example usage
base_prompt = "Describe a successful entrepreneur."
fair_prompt = create_fairness_aware_prompt(base_prompt)
```

## Conclusion: Building More Equitable AI Systems

By applying these techniques with modern language models, you can:

1. Identify biases more effectively through counterfactual analysis
2. Ensure fair representation through stratified sampling
3. Implement fairness constraints to mitigate biases
4. Compare performance across different demographic groups
