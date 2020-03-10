# Knock out rule optimisation using Non dominated Sorting Genetic Algorithm-II

Knock out Rules (KOR) are generally the first step in evaluating a credit applications. 
KOR were developed in the pre-machine learning era where institutins developed expert based scorecards for scoring credit applications. This means starting off with simple set of KOR such as e.g. no applicant
under 18 is accepted, no applicant who is unemployed is accepted etc. 
However these become limiting in the age of ML, where you would ideally want to score all applicants based on data that you have and not reject them based on KOR.

Within financial organisations there is ever growing demand to optimise these knockout rules i.e. only keep the KOR which are mandatory and relax or eliminiate rules which are not adding benefits . At the same time while relaxing the rules it is important to keep the default rate low .

From an optimisation prespective this can be formulated as multi objective optimisatino problem,where the objectives are :
*  Reduce the no.of KOR.
*  Maintain high AUC of the model.
*  Increase the no.of applications.

