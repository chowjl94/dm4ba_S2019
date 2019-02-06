import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pylab as plt


# A function that picks a color for an instance, depending on its target variable
# We use 0 for "No" and "1" for yes.
# The function returns a list of items, one item for each instance (in the order given)
def Color_Data_Points(target):
    color = ["red" if t == 0 else "blue" for t in target]
    return color


def get_target_name(trg=None):
    if trg is None:
        return "response"
    return trg


def create_data(n_points, d_features=2, features=None, target_name=None):
    '''
    This function creates a data set with 3 random normal distributions scaled. 
    There are two main variables in this dataset: humor and number_pets.
    It also computes higher orders for the 'humor' variable (^2, ^3 and ^4).
    You can change this function with new orders or column names.
   
    RETURNS: target_name ("response", if not provided), 
             feat_names (list with the feature names. If not explicitly given,
             the list contains ["feat_1", "feat_2", ...]), 
             data  (dataframe with the data we want to compute), 
             Y (target variable with values 0 or 1)
    '''

    # Relationships
    feat_names = features  # ["humor", "number_pets"]
    if feat_names is None:
        feat_names = [ "feat_" + str(i+1) for i in range(0, d_features) ]

    target_name = get_target_name(target_name)

    # Generate data (4 random normal distributions!!!!)
    a = np.random.normal(5, 10, n_points )
    b = np.random.normal(25, 18, n_points )
    c = np.random.normal(35, 15, n_points )
    d = np.random.normal(6, 10, n_points )

    # Change scales
    x2 = list(a+10) + list(a+10) + list(b+10) + list(a+30)+ list(d+10)+ list(a+10)
    x1 = list((d+10)/10) + list((b+60)/10) + list((c+10)/10) + list((d+10)/10)+ list((c+10)/10)+ list((c+10)/10)

    target = list(np.zeros(len(b))) + list(np.ones(len(b))) + list(np.zeros(len(b)))+ list(np.ones(len(b)))+ list(np.zeros(len(b)))+ list(np.zeros(len(b)))

    #predictors, target = datasets.make_classification(n_features=2, n_redundant=0, 
    #                                              n_informative=n_informative, n_clusters_per_class=2,
     #                                             n_samples=n_users)
    
    data = pd.DataFrame(np.c_[x1, x2], columns=feat_names)
    #data = pd.DataFrame(predictors, columns=variable_names)

    # Add interactions
    first_feat_name = feat_names[0]
    for p in range(2, 7):
        data[first_feat_name + '^' + str(p)] = np.power(data[first_feat_name], p)

    data[target_name] = Y = target
    Y = data[target_name]
    return target_name, feat_names, data, Y



def Decision_Surface(data, target, model=None, surface=True, probabilities=True, cell_size=.01):
    '''
    This function creates the surface of a decision tree using the data created with this script. 
    You can change this function tu plot any column of any dataframe. 
    
    INPUT: data (created with data_tools.X() ),
            target (Y value creted with data_tools.create_data() ),
            model (Model already fitted with X and Y , i.e. DecisionTreeClassifier or logistic regression )
            surface (True if we want to display the tree surface),
            probabilities (False by default, if True we can see the color-scale based on the likelihood of being closer to the separator),
           cell_size (value for the step of the numpy arange that creates the mesh)
    RETURNS: Scatterplot with/without the surface
    '''
    # Get bounds, we only have 2 columns in the dataframe: column 0 and column 1 
    x_min, x_max = data[data.columns[0]].min(), data[data.columns[0]].max()
    y_min, y_max = data[data.columns[1]].min(), data[data.columns[1]].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_size), np.arange(y_min, y_max, cell_size))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    # Add interactions
    for i in range(data.shape[1]):
        if i <= 1:
            continue
        meshed_data = np.c_[meshed_data, np.power(xx.ravel(), i)]

    # Plot mesh and data
    plt.title("{} and {}".format(data.columns.values[0], data.columns.values[1]))
    plt.xlabel(data.columns.values[0])
    plt.ylabel(data.columns.values[1])

    if surface and model != None:

        levels=[-1,0,1]

        # Predict on the mesh with labels or probability
        if probabilities:
            # Color-scale on the contour (surface = separator)
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.4)
        else:
            # Only a curve/line on the contour (surface = separator)
            Z = model.predict(meshed_data).reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.coolwarm_r, alpha=0.4)

    color = Color_Data_Points(target)
    plt.scatter(data[data.columns[0]], data[data.columns[1]], color=color, edgecolor='black')
    return


def X(data, complexity=1):
    '''
    This function return the X-data from the 'create_data' function of this script.
    You can change the complexity to receive the main 2 columns + complex orders.
    
    INPUT: complexity (higher complexity (1 to 4) for the 'humor' variable)
    RETURNS: data  (dataframe with the data WITH higher orders IF required)
    '''
    # remove the target variable
    drops = [get_target_name()]
    
    # if complexity = 1 then we just need to drop all the higher order from the dataframe
    # based on the number of complexity required, we drop the rest of the higher orders
    first_feat_name = data.columns.values[0]
    for p in range(complexity+1, 7):
        drops.append(first_feat_name + "^" + str(p))
    
    return data.drop(drops, 1)

# target_name, variable_names, data, Y = create_data()