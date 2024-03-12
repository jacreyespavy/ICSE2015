# ICSE2025



Project Structure：
│ ├── ChatGPT/ 
│ ├────── outputs/ # The results of different insight prompts in the experiment
│ ├────── gpt_completion.py # How to configure ChatGPT API
│ ├── human_labeled/  # Test set and its manual annotation
│ ├── input/  # The pure texts of test set
│ ├── Screenshot of conversation/  # Some screenshots show how we interact with AIGC tools
│ ├── analysis_senti_for_file.py # Main program entry, used to set which prompt to complete the SA4SE tasks
│ ├── data analysis.xlsx # Detailed evaluation data for all experiments
│ ├── evaluate.py 
│ ├── explainer.py # Using a TF-IDF based logistic regression model to fit ChatGPT predictions for interpretation
│ ├── evaluate.py 
│ ├── prompts.py # The main file for storing insight-enhanced prompts
