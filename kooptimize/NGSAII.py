import random as rn
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import math
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
# For use in Jupyter notebooks only:

# Create reference solutions
# --------------------------


def get_ko_rule_dict():
    rule_dict = {
        1:"age_when_disbursal <= 840 ",
        2:"ltv <= 95",
        3:"delinquent_accts_in_last_six_months <=30",
        4:"pri_no_of_accts <= 20",
        5:"sec_no_of_accts <= 35",
        6:"avg_acct_age_m >= 6",
        7:"new_accts_in_last_six_months <= 10",
        8:"perform_cns_score >= 200",
        9:"pri_overdue_accts <=10",
        10:"amt_rejected <= 0.7"}

    return rule_dict

def get_rules_for_individual(individual):
    
    selected_rules_dict = {}
    rule_dict = get_ko_rule_dict()
    rule_key_list = list(rule_dict.keys())
    assert len(individual)==len(rule_key_list),"Individual lenght != Rule Lenght"
    
    valid_rules = [rule_key_list*individual for rule_key_list,individual in 
                   zip(rule_key_list,individual)]
    
    valid_rules = [key for key in valid_rules if key >=1]
    
    for key in valid_rules :
        selected_rules_dict[key]=rule_dict[key]

    
    return selected_rules_dict

def apply_rules_to_df(X,y,selected_rules):
    
    df = X
    df['target'] = y
    
    if len(selected_rules.keys())>=1:
        for key in selected_rules.keys():
            df = df.query(selected_rules[key]).copy()
        y = df['target']
        X = df.drop(['target'],axis=1)
        return X,y
    else :
        return X,y


def get_trained_model(X,y):
    """
    Returns a trained model.
    """
#     clf = GradientBoostingClassifier(n_estimators=500,max_features='auto',
#                                      learning_rate=0.01,random_state=0)
    clf = GradientBoostingClassifier(max_features='auto')
    score = cross_validate(clf,X,y,scoring=['roc_auc'],cv=5)
    return clf.fit(X,y),(np.mean(score['test_roc_auc'])*100)

    
def show_population_score_df(population,scores,
column_name=['#Rules','AUC Diff','No.ofApplications']):
    
    result_df = pd.DataFrame(scores,columns=column_name)
    #result_df = pd.DataFrame(scores,columns=['AUC','No.ofApplications'])
    result_df = result_df.apply(lambda x: np.abs(x) if x.name in [column_name[0],column_name[1]] else x)
    #Add Rules to dataframe
    result_df['Rules'] = [get_rules_for_individual(individual) for individual in population]
    #Sort dataframe by AUC Difference
    result_df.sort_values(by=column_name[1],ascending=True,inplace=True)
    
    return result_df


def create_reference_solutions(chromosome_length, solutions):
    """
    Function to create reference chromosomes that will mimic an ideal solution
    """
    references = np.zeros((solutions, chromosome_length))
    number_of_ones = int(chromosome_length / 2)

    for solution in range(solutions):
        # Build an array with an equal mix of zero and ones
        reference = np.zeros(chromosome_length)
        reference[0: number_of_ones] = 1

        # Shuffle the array to mix the zeros and ones
        np.random.shuffle(reference)
        references[solution, :] = reference

    return references

def get_cv_score(df,Y,feature_list=[],shuffle=True):
    """
    :param df: dataframe
    :param target_col: Target Column train the model on .
    :param feature_list: List of features to work with .
    :return: scoring score
    """
    # Always select the last column ie the target.

    #feature_list[len(feature_list)-1] = 1
    selected_rules = get_rules_for_individual(feature_list)
    X_ko,y_ko = apply_rules_to_df(df,Y,selected_rules)

    array = X_ko.values
    X = array[:,:].astype(float)

    if(shuffle):
        seed = random.randint(0,df.shape[0])
        sss = StratifiedShuffleSplit(n_splits=10,train_size=0.10,random_state=seed)
        train_index, test_index = next(sss.split(X,y_ko))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_ko[train_index], y_ko[test_index]
    else:
        X_train = X
        y_train = Y

    parameters = {'learning_rate': [0.01,0.02,0.05]}
    clf = GradientBoostingClassifier(max_features='auto')
    random_search = RandomizedSearchCV(clf,param_distributions=parameters,
    n_iter=10,cv=3,scoring='roc_auc',iid=False)
    random_search.fit(X,y_ko)
    return random_search.best_score_*100

def get_auc(X,y,trained_model,individual):
    """
    Calculate the AUC for given individual from the trained model.
    """
    selected_rules = get_rules_for_individual(individual)
    X_ko,y_ko = apply_rules_to_df(X,y,selected_rules)
    y_pred = trained_model.predict_proba(X_ko)[:,1]
    return roc_auc_score(y_ko,y_pred)*100



def calculate_fitness(reference, population):
    """
    Calculate how many binary digits in each solution are the same as our
    reference solution.
    """
    # Create an array of True/False compared to reference
    identical_to_reference = population == reference
    # Sum number of genes that are identical to the reference

    fitness_scores = identical_to_reference.sum(axis=1)
    return fitness_scores


def score_population(df,Y,population):
    """
    Loop through all reference solutions and request score/fitness of
    populaiton against that reference solution.
    """

    del_list = []

    for index in range(population.shape[0]):
        if np.sum(population[index]) == 0:
            del_list.append(index)

    population = np.delete(population, del_list, axis=0)

    scores = np.zeros((population.shape[0],3))
    # Minimize function 1
    scores[:,0] = -1*population.sum(axis=1)

    auc_list = []
    no_of_records_list = []

    for individual in population:
        #print("Individual",indvidual)
        #individual[len(individual)-1]=0
        if np.sum(individual)==0:
            print("Empty Individual found ! Check removal code")
            auc_list.append(0)
        else:
            auc = get_cv_score(df,Y,individual)
            #auc = get_cv_score_lr(df,Y,individual)
            auc_list.append(auc)
            X_ko,y_ko=apply_rules_to_df(df,Y,get_rules_for_individual(individual))
            no_of_records = X_ko.shape[0]
            #auc_complexity_score = get_combined_score(no_of_features,auc)
            #print(individual,auc,no_of_records)
            no_of_records_list.append(no_of_records)

    #Maximize function 2
    scores[:,1] = auc_list
    #Maximize function 2
    scores[:,2] = no_of_records_list
    #print("Scores",scores)
    return population,scores

def score_population_with_trained_clf(df,Y,clf,population):
    """
    Loop through all reference solutions and request score/fitness of
    populaiton against that reference solution.
    """

    del_list = []

    for index in range(population.shape[0]):
        if np.sum(population[index]) == 0:
            del_list.append(index)

    population = np.delete(population, del_list, axis=0)

    scores = np.zeros((population.shape[0],3))
    # Minimize function 1
    scores[:,0] = -1*population.sum(axis=1)

    auc_list = []
    no_of_records_list = []

    for individual in population:
        #print("Individual",indvidual)
        #individual[len(individual)-1]=0
        if np.sum(individual)==0:
            print("Empty Individual found ! Check removal code")
            auc_list.append(0)
        else:
            auc = get_auc(df,Y,clf,individual)
            base_auc_diff = auc-67
            #auc = get_cv_score_lr(df,Y,individual)
            auc_list.append(base_auc_diff)
            X_ko,y_ko=apply_rules_to_df(df,Y,get_rules_for_individual(individual))
            no_of_records = X_ko.shape[0]
            #total_amount = np.log(np.sum(X_ko['disbursed_amount']))
            #print(np.sum(individual),base_auc_diff,no_of_records)
            no_of_records_list.append(no_of_records)

    #Minimize function 2
    scores[:,1] = auc_list
    #Maximize function 2
    scores[:,2] = no_of_records_list
    #print("Scores",scores)
    return population,scores

# Calculate crowding and select a population based on crowding scores
# -------------------------------------------------------------------

def calculate_crowding(scores):
    """
    Crowding is based on a vector for each individual
    All scores are normalised between low and high. For any one score, all
    solutions are sorted in order low to high. Crowding for chromsome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores
    """

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum croding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def reduce_by_crowding(scores, number_to_select):
    """
    This function selects a number of solutions based on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher croding dostance is always picked
    """
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1ID = rn.randint(0, population_size - 1)

        fighter2ID = rn.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[
            fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[
                fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID),
                                       axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)

            crowding_distances = np.delete(crowding_distances, (fighter1ID),
                                           axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]

            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)

            crowding_distances = np.delete(
                crowding_distances, (fighter2ID), axis=0)

    # Convert to integer
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)

    return (picked_population_ids)


# Pareto selecion
# ---------------

def identify_pareto(scores, population_ids):
    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected solutions.
    """

    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def build_pareto_population(
        population, scores, minimum_population_size, maximum_population_size):
    """
    As necessary repeats Pareto front selection to build a population within
    defined size limits. Will reduce a Pareto front by applying crowding
    selection as necessary.
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front = identify_pareto(
            scores[unselected_population_ids, :], unselected_population_ids)

        # Check size of total parteo front.
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_individuals = (reduce_by_crowding(
                scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]

        # Add latest pareto front to full Pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        # Update unslected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    population = population[pareto_front.astype(int)]
    return population


# Population functions
# --------------------

def create_population(individuals, chromosome_length):
    """
    Create random population with given number of individuals and chroosome
    length.
    """

    # Set up an initial array of all zeros
    population = np.zeros((individuals, chromosome_length))
    # Loop through each row (individual)
    for i in range(individuals):
        # Choose a random number of ones to create
        ones = rn.randint(1, chromosome_length)
        # Change the required number of zeros to ones
        population[i, 0:ones] = 1
        # Sfuffle row
        np.random.shuffle(population[i])

    return population


def breed_by_crossover(parent_1, parent_2):
    """
    Combine two parent chromsomes by crossover to produce two children.
    """
    # Get length of chromosome
    chromosome_length = len(parent_1)

    # Pick crossover point, avoding ends of chromsome
    crossover_point = rn.randint(1, chromosome_length - 1)

    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))

    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))

    # Return children
    return child_1, child_2


def randomly_mutate_population(population, mutation_probability):
    """
    Randomly mutate population with a given individual gene mutation
    probability. Individual gene may switch between 0/1.
    """

    # Apply random mutation
    random_mutation_array = np.random.random(size=(population.shape))

    random_mutation_boolean = \
        random_mutation_array <= mutation_probability

    population[random_mutation_boolean] = \
        np.logical_not(population[random_mutation_boolean])

    # Return mutation population
    return population


def breed_population(population):
    """
    Create child population by repetedly calling breeding function (two parents
    producing two children), applying genetic mutation to the child population,
    combining parent and child population, and removing duplice chromosomes.
    """
    # Create an empty list for new population
    new_population = []
    population_size = population.shape[0]
    # Create new popualtion generating two children at a time
    for i in range(int(population_size / 2)):
        parent_1 = population[rn.randint(0, population_size - 1)]
        parent_2 = population[rn.randint(0, population_size - 1)]
        child_1, child_2 = breed_by_crossover(parent_1, parent_2)
        new_population.append(child_1)
        new_population.append(child_2)

    # Add the child population to the parent population
    # In this method we allow parents and children to compete to be kept
    population = np.vstack((population, np.array(new_population)))
    population = np.unique(population, axis=0)

    return population


# Plot Pareto front (for two scores only)
def plot_2d_paretofront(scores,generation,labels):
    """
    Function to plot the 2D plot of
    :param scores: scores to plot
    :return:
    """
    x = scores[:, 0]*-1
    y = scores[:, 1]*-1
    z = scores[:, 2]
    plt.rcParams["figure.figsize"]=(15,12)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    
    ax1.scatter(x,y)
    ax1.set(xlabel=labels[0],ylabel=labels[1])

    ax2.scatter(y,z)
    ax2.set(xlabel=labels[1],ylabel=labels[2])

    ax3.scatter(x,z)
    ax3.set(xlabel=labels[0],ylabel=labels[2])
    
    fig.suptitle("NSGA 2 : Generation {}".format(generation),fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('images/pareto_gen_{}.png'.format(generation))
    plt.show()


def plot_3d_paretofront(scores,generation,labels) :
    X = list(scores[:,0]*-1)
    Y = list(scores[:,1]*-1)
    Z = list(scores[:,2])

    plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),\
                               np.linspace(np.min(Y),np.max(Y),10))
    plotz = interp.griddata((X,Y),Z,(plotx,ploty),method='linear')

    fig = plt.figure(figsize=(10,10))
    fig.suptitle("NSGA 2 : Generation {}".format(generation),fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel=labels[0],ylabel=labels[1],zlabel=labels[2])
    ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis', edgecolor='none')
    fig.savefig('images/pareto_3D_gen_{}.png'.format(generation))
    plt.show() 

def plot_gen_time(generations,time_taken):
    """
    """
    fig,ax = plt.subplots()
    ax.plot(generations,time_taken)
    ax.set(xlabel="Generations",ylabel="Time (secs)")
    ax.set_title('Time required per Generation.')
    plt.show()
