'''
CS5800 Fall 2023 Group Project Juliette Kamtamneni, David Riggle, Mitchell Hornsby

This program creates a data structure to cluster hypothetical children's skills and
the age they were attained in order to cluster and predict next skills to obtain
given the assigned cluster as well as simple determination if the child is ahead or
behind and areas to focus on.

This program requires installation of numpy, scipy, matplotlib in the environment.
'''

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import random
from collections import Counter


# Step 1: Generate Mock Data
def generate_mock_data(num_children, num_skills, max_age):
    '''
    This function generates an array of mock data for children, skills and age of attainment
    :param num_children: integer for number of children to compare
    :param num_skills: integer for number of skills to compare
    :param max_age: integer for max age of children in data generation
    :return: list of mock data for each child, their skills and age acquired (0 is not attained)
    '''
    mock_data = []
    for _ in range(num_children):
        child_data = {'ChildID': _, 'Skills': {}}
        for skill_id in range(num_skills):
            age = random.randint(0, max_age)
            child_data['Skills'][f'Skill_{skill_id + 1}'] = age
        mock_data.append(child_data)
    #print('mock_data...\n', mock_data)
    return mock_data


# Step 2: Perform Hierarchical Single Link Clustering and Create a Dendrogram
def perform_clustering(mock_data):
    '''
    This function calls the mock data structure and performs single link clustering
    using euclidean distance to group the children by skills attained
    :param mock_data: list generated in Step 1
    :return: returns the linkage matirx used to create dendrogram for analysis
    '''
    skill_data = np.array([list(child['Skills'].values()) for child in mock_data])
    linkage_matrix = linkage(skill_data, method='single', metric='euclidean')
    dendrogram(linkage_matrix)
    plt.xlabel('Children')
    plt.ylabel('Distance')
    plt.title('Hierarchical Single-Link Clustering Dendrogram')
    #plt.show()
    plt.savefig('dendogram_sk.png')
    #print('linkage_matrix...\n', linkage_matrix)
    return linkage_matrix


# Step 3a: Compare Child's Skill Acquisition Timeline with Cluster Patterns
def compare_child_to_clusters(child_data, mock_data, linkage_matrix, threshold=8.5):
    '''
    This function compares an individual selected child to the cluster and determines
    by simple average age if the child is advanced, typical or delayed
    :param child_data: list of a single selected child's data
    :param mock_data: list of entire mock data strucutre from step 1
    :param linkage_matrix: linkage cluser matrix from step 2
    :param threshold: finely tuned threshold based on cluster size to balance size of clusters versus
    have whole clusters with exactly the same skills or single sized clusters
    :return: statement on the child's development
    '''
    # Search the cluster for given child data
    child_skills = np.array(list(child_data['Skills'].values()))
    child_cluster_label = fcluster(linkage_matrix, threshold, criterion='distance')[child_data['ChildID']]
    cluster_children_skills = [np.array(list(child['Skills'].values()))
                               for child in mock_data if fcluster(
                                    linkage_matrix, threshold, criterion='distance')[child['ChildID']] == child_cluster_label]


    # Calculate mean age of skill acquisition for the cluster
    cluster_mean_age = np.mean([np.mean(child_skills[child_skills > 0]) for child_skills in cluster_children_skills])

    print('Mean age child skills:', np.mean(child_skills[child_skills > 0]))
    print('Cluster mean age:', cluster_mean_age)

    if np.mean(child_skills[child_skills > 0]) < cluster_mean_age:
        return "Advanced"
    elif np.mean(child_skills[child_skills > 0]) > cluster_mean_age:
        return "Delayed"
    else:
        return "Typical"


# Step 3b: Predict the Next Skill
def predict_next_skill(child_data, mock_data, linkage_matrix, threshold=8.5):
    '''
    This function compares an individual selected child to the cluster and determines
    what skills are missing for the selected child versus the cluster and predicts
    those as skills to attain next
    :param child_data: list of a single selected child's data
    :param mock_data: list of entire mock data strucutre from step 1
    :param linkage_matrix: linkage cluster matrix from step 2
    :param threshold: finely tuned threshold based on cluster size
    :return: missing skills for the child from the cluster they reside in
    '''
    # Search the cluster for given child data
    child_cluster = fcluster(linkage_matrix, threshold, criterion='distance')
    cluster_children = [child for child in mock_data if
                        fcluster(linkage_matrix, threshold, criterion='distance')[child['ChildID']] == child_cluster[child_data['ChildID']]]

    # Find skills that the target child has not acquired
    target_child_missing_skills = {skill for skill, age in child_data['Skills'].items() if age == 0}

    # Find skills that other children in the same cluster have acquired
    acquired_skills_in_cluster = set()
    for other_child in cluster_children:
        if other_child['ChildID'] != child_data['ChildID']:
            acquired_skills = {skill for skill, age in other_child['Skills'].items() if age > 0}
            acquired_skills_in_cluster.update(acquired_skills)

    # Determine which acquired skills in the cluster the target child is missing
    missing_skills_for_target_child = acquired_skills_in_cluster.intersection(target_child_missing_skills)

    return list(missing_skills_for_target_child)


# Step 4a: Recommend Activities Based on Current Skillset and Developmental Pathways
def recommend_activities(child_data, mock_data, linkage_matrix, threshold=8.5):
    '''
    This function compares an individual selected child to the cluster and determines
    what skills are missing for the selected child versus the cluster and recommends skills
    to focus on to catch up or further exceed their cluster.  Could be enhanced with out of sample
    time series which was out of scope due to time constraints.
    :param child_data: list of a single selected child's data
    :param mock_data: list of entire mock data strucutre from step 1
    :param linkage_matrix: linkage cluster matrix from step 2
    :param threshold: finely tuned threshold based on cluster size
    :return: missing skills for the child as areas recommended to focus on
    '''
    child_cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    target_child_cluster_label = child_cluster_labels[child_data['ChildID']]

    cluster_children = [child for child in mock_data if child_cluster_labels[child['ChildID']] == target_child_cluster_label]

    # Find skills that the target child has not acquired (i.e., skill value is 0)
    target_child_missing_skills = {skill for skill, age in child_data['Skills'].items() if age == 0}

    # Find skills that other children in the same cluster have acquired
    acquired_skills_in_cluster = set()
    for other_child in cluster_children:
        if other_child['ChildID'] != child_data['ChildID']:
            acquired_skills = {skill for skill, age in other_child['Skills'].items() if age > 0}
            acquired_skills_in_cluster.update(acquired_skills)

    # Determine which acquired skills in the cluster the target child is missing
    missing_skills_for_target_child = acquired_skills_in_cluster.intersection(target_child_missing_skills)

    if missing_skills_for_target_child:
        recommendations = [f"Activity to develop {skill}: ..." for skill in missing_skills_for_target_child]
        return list(recommendations)
    else:
        return "No recommendations available"


# Step 4b: Provide Tailored Activity Recommendations for Advanced or Delayed Development
def tailored_activity_recommendations(comparison_result):
    '''
    This function calls the result of the child comparison if the child is advanced, typical or delayed
    and suggests activity recommendations.  Could be enhanced with out of sample data and real expected
    skills progression
    :param comparison_result: string containing child's designation category
    :return: string with suggested activities
    '''

    if comparison_result == "Advanced":
        return ["Your child's development is advanced. Consider challenging activities in areas of interest."]
    elif comparison_result == "Delayed":
        return ["Your child's development is delayed. Focus on activities to catch up in critical skills."]
    else:
        return ["Your child's development is typical.  Focus on reinforcing crtical skills or challenging activities"]


def main():
    # Generate mock data
    num_children = 100
    num_skills = 20
    max_age = 6
    mock_data = generate_mock_data(num_children, num_skills, max_age)

    # Perform clustering and create dendrogram
    linkage_matrix = perform_clustering(mock_data)

    # Check threshold size for clusters if changing num_children, num_skills or max_age parameters above
    '''
    for threshold in np.linspace(start=1.0, stop=10, num=50):  # Adjust range and step as needed
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        cluster_sizes = Counter(cluster_labels)
        print(f"Threshold: {threshold}, Cluster Sizes: {cluster_sizes}")
    '''

    # Example child data for analysis
    child_data = mock_data[0]
    #print('main child_data...\n', child_data)

    # Show cluster sizes from dendrogram
    cluster_labels = fcluster(linkage_matrix, 8.5, criterion='distance')
    cluster_sizes = Counter(cluster_labels)
    print(f"Cluster Sizes: {cluster_sizes}")

    # Step 3a: Compare Child's Skill Acquisition Timeline with Cluster Patterns
    comparison_result = compare_child_to_clusters(child_data, mock_data, linkage_matrix)
    print(f"Child's development is {comparison_result}.")

    # Step 3b: Predict the Next Skill
    next_skill = predict_next_skill(child_data, mock_data, linkage_matrix)
    print(f"Predicted next skill for the child: {next_skill}")

    # Step 4a: Recommend Activities Based on Current Skillset and Developmental Pathways
    activity_recommendations = recommend_activities(child_data, mock_data, linkage_matrix)
    # print('activity_recommendations...\n', activity_recommendations)
    print(f"Next recommendation for child: {activity_recommendations}")

    # Step 4b: Provide Tailored Activity Recommendations for Advanced or Delayed Development
    tailored_recommendations = tailored_activity_recommendations(comparison_result)
    # print('tailored_recommendations...\n', tailored_recommendations)
    for recommendation in tailored_recommendations:
        print(recommendation)


if __name__ == "__main__":
    main()