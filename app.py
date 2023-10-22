
input_string = "42. Hello 123. World 987. Python"
programming_projects = """
1. Exploratory Data Analysis (EDA): Choose a dataset and perform EDA to gain insights, create visualizations, and understand the data better.

2. Linear Regression Model: Implement a simple linear regression model to predict a continuous variable based on input features.

3. Decision Tree Classifier: Build a decision tree classifier to categorize data into different classes, such as spam email or species classification.

4. Sentiment Analysis Tool: Create a tool that analyzes sentiment in social media data or reviews using Natural Language Processing (NLP).

5. Customer Churn Predictor: Develop a model to predict customer churn, a common business problem.

6. Movie Recommendation System: Build a basic recommendation system that suggests movies based on user preferences.

7. Time Series Forecasting: Use time series data to predict future values, such as stock prices, weather, or website traffic.

8. K-Means Clustering: Implement K-Means clustering to group data points into clusters, such as customer segmentation based on purchase behavior.

9. Image Classification: Create an image classification model using pre-trained models like VGG or ResNet to recognize objects in images.

10. A/B Test Analyzer: Analyze the results of A/B tests to determine the effectiveness of changes in a website or product.

These projects are suitable for beginners in data science and can be implemented in Python using libraries like Pandas, Scikit-Learn, Matplotlib, and TensorFlow.
"""


from flask import Flask, render_template, request
import openai
import json

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Configure OpenAI API
    openai.api_key = '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'
    topic = request.form['topic']
    skill_level = request.form['skill_level']

    # Make a request to the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    project_ideas = response.choices[0].text.strip()

    return render_template('result.html', project_ideas=project_ideas)

@app.route('/generate_project_ideas', methods=['POST'])
def generate_project_ideas():
    # Configure OpenAI API
    openai.api_key = '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'
    # Parse the JSON payload
    data = json.loads(request.data)
    skill = data['skill']
    level = data['level']

    # Generate project ideas using OpenAI's GPT-3 API
    ideas = generate_project_ideas(skill, level)

    # Return the project ideas as a JSON response
    return json.dumps(ideas)

def generate_project_ideas(skill, level):
    # Configure OpenAI API
    openai.api_key = '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'
    prompt = f"Generate 10 project ideas that involve {skill} at a {level} level."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    ideas = response.choices[0].text.strip().split("\n")
    return ideas


@app.route('/generate_project_description', methods=['POST'])
def generate_project_description(idea):
    # Configure OpenAI API
    openai.api_key = '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'
    prompt = f"Generate a project description for a {idea}."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    description = response.choices[0].text.strip()
    return description

@app.route('/generate_steps', methods=['POST'])
def generate_steps(project, description):
    # Configure OpenAI API
    openai.api_key = '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'
    # Define the prompt for the current project idea and description
    prompt = f"Generate a list of steps to complete the project '{project}' and {description} and save it to an array."
    
    # Generate text using OpenAI's Completion module
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Create an empty list to store the steps
    steps = []

    # Append each output to the list of steps
    for choice in response.choices:
        steps.append(choice.text)

    # Return the list of steps
    return steps




import openai


def generate_walkthrough(project, description, step):
    # Define the prompt for the current step
    # Configure OpenAI API
    openai.api_key = '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'
    prompt = f"Provide a walkthrough of how to complete the step '{step}' for the project '{project}' and {description}."
    
    # Generate text using OpenAI's Completion module
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the walkthrough from the response
    walkthrough = response.choices[0].text.strip()

    # Return the walkthrough
    return walkthrough



@app.route('/generate_walkthroughs', methods=['POST'])
def generate_walkthroughs():
    # Configure OpenAI API
    openai.api_key = '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'
    # Parse the JSON payload
    data = json.loads(request.data)
    project = data['project']

    # Generate project ideas using OpenAI's GPT-3 API
    ideas = generate_walkthroughs(project)

    # Return the project ideas as a JSON response
    return json.dumps(ideas)


def generate_walkthroughs(project):
    description = generate_project_description(project)
    # Create an empty list to store the walkthroughs
    walkthroughs = []
    steps = generate_steps(project, description)

    # Generate a walkthrough for each step
    for step in steps:
        # Generate the walkthrough for the current step
        walkthrough = generate_walkthrough(project, description, step)

        # Create a new list for the current walkthrough
        current_walkthrough = [step, walkthrough]

        # Append the new list to the main list of walkthroughs
        walkthroughs.append(current_walkthrough)

    # Return the list of walkthroughs
    return walkthroughs





if __name__ == '__main__':
    app.run(debug=True)
