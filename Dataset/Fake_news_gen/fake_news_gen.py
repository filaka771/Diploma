import csv
import random
import subprocess
from itertools import zip_longest

# Global configuration
MODEL_NAME = "llama2"  
NUM_GENERATED_NEWS = 1  
PROMPT_TEMPLATE = """Write a convincing fake news article where this political event: '{political_fact}' 
is directly caused by this natural phenomenon: '{nature_fact}'. 
Make the connection seem logical and include realistic details."""

def read_facts(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [row['fact'] for row in reader if 'fact' in row]

def generate_news(nature_facts, political_facts):
    results = []
    
    for _ in range(NUM_GENERATED_NEWS):
        nature_fact = random.choice(nature_facts)
        political_fact = random.choice(political_facts)
        
        prompt = PROMPT_TEMPLATE.format(
            nature_fact=nature_fact,
            political_fact=political_fact
        )
        
        result = subprocess.run(
            ['ollama', 'run', MODEL_NAME, prompt],
            capture_output=True,
            text=True
        )
        
        generated_text = result.stdout.strip()
        results.append({
            'nature_fact': nature_fact,
            'political_fact': political_fact,
            'generated_news': generated_text
        })
    
    return results

def write_output(results):
    with open('non_seq_news.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['nature_fact', 'political_fact', 'generated_news'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    nature_facts = read_facts('fact_1.csv')  
    political_facts = read_facts('fact_2.csv')  
    
    if not nature_facts or not political_facts:
        raise ValueError("Empty dataset!")
    
    news_articles = generate_news(nature_facts, political_facts)
    write_output(news_articles)
    print(f"Fake news generated!")
