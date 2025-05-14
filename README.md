This project is a Python-based tool for analyzing early childhood skill acquisition using hierarchical clustering. It models typical developmental patterns, identifies delays or advanced progress, and recommends personalized activities to support the "next" skill each child is ready to learn.

🧠 Project Overview
Goal:
To create a data-driven program that tracks child development, identifies common clusters of emerging skills, and uses those patterns to provide targeted developmental insights and recommendations.

Key Features:

🧩 Generates mock longitudinal data simulating developmental milestones for 100+ children.

🔗 Performs single-link hierarchical clustering using SciPy to identify patterns in skill acquisition.

🧮 Compares individual timelines to group clusters to detect advanced or delayed development.

🎯 Predicts likely next skills for a child based on historical cluster behavior.

📝 Recommends activities tailored to each child’s current development stage.

🛠 Technologies Used
Python 3

SciPy – Hierarchical clustering (scipy.cluster.hierarchy)

NumPy – Numerical analysis

Matplotlib – Dendrogram and data visualization

Random – Simulated mock data generation

📊 Developmental Milestones
The program uses 25 common early childhood milestones (e.g., Crawling, Walking, Talking, etc.), and generates randomized acquisition ages per child to simulate real-world variation. A time-series option tracks changes across multiple stages.

The program will:

Generate mock child development data

Visualize a dendrogram of clustered development patterns

Output predictions and recommendations for a selected child

This is a collaborative project developed by graduate students : Mitchell Hornsby, Juliette Kamtamneni, and David Riggle as part of:
Course: Algorithms CS5800
Professor: Justin Kennedy
December 3, 2023
