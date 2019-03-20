# sdm_kmeans
Scientific Data Management Assignment 1 KMeans Clustering

## Notes for Codebase
firstly run python setup.py install
Then run pip install -r requirements.txt to install dependencies
Recommended style guide: https://www.python.org/dev/peps/pep-0008/


## Way of approaching tasks
1. Everyone writes someone elses test cases (Lorenz: Good idea but project too small IMO)
2. Everyone hardcodes expected outputs of their domain
3. Everyone takes the inputs and works on developing an output testing against test caes
4. Split the work into update and init algorithms and we work on that..
5. Train the models and tweak algorithm
6. Test and improve
7. Run against test set
8. Upload

## TEAM GOAL PRIORITIES
* Implement basic testcases
* implement a fully working script with defined K, random initialisation and Lloyds.
* Implement Mac Queen Update & furthest point technique
* Implement 1 other pre-clustered sample initialisation technique

## Team TODO List
STRATEGY PATTERN TODO:
* Create 2 concrete classes for the update_strategy (1 hr)
* Create 3 concrete classes for init_strategy (1 hr) (Lorenz: RandomInit and FarthestPointInit already implemented, 1 more to go)
* Update kmeans.py (context class) to implement this (1 hr)
* decide on which 2 initialisation strategies we use (2 hrs) (Already decided: FarthestPointInit and PreClusteredSampleInit)

KMEANS TODO
1. Write testcases (ALL) (1hr each)
    Divide up testcases````
    Write outputs for functions (ALL)

2. Import data from txt and csv filetype into a pandas dataframe (20 mins) (Lorenz: Done)

3. Clean data if necessary? (Unnecessary)
4. display a summary of data (20 mins)
    AND if k_clusters isn't defined then find optimal number for K (might not be necessary?) (2 hrs)
5. Split data into training and test set potentially creating a case for time series
    but can probably just leave that for now. (20 mins)
6. Implement each initialisation strategy (3 hrs each) (Lorenz: RandomInit and FarthestPointInit already implemented, 1 more to go. Workload estimate more like 6 hrs each IMO)
8. Implement each update strategy (3 hrs each)
9. Print our result and a visual representation of it. (30 mins)
10. Improve our results (~)
11. Submit! (30 mins)


## TO FIND OUT
1. Will K always be given or do we have to find out. Asnwear = It s labeled data, skinf or no skin
2. Is speed important, should we be concerned with parallelisation? 
3. Points in space or points of data? (I think points of data)
4. Does the data need to be cleaned

