import shap

nn_model = loaded_model
background = X_train_wo[:25]

def shap_explainer(model_type, model, background, X_train_wo):

    if model_type == 'nn':
        explainer = shap.KernelExplainer(model=nn_model.predict, data=background)

    elif model_type == 'rf':
        explainer = shap.TreeExplainer(random_forest)


    explainer = shap.KernelExplainer(model=nn_model.predict, data=background)
    shap_values = explainer.shap_values(background)
    instance_index = 5
    rounded_shap_values = np.round(shap_values[0][instance_index], 1)
    rounded_features = np.round(X_train_wo.iloc[instance_index], 1)
    shap.initjs()
    shap.force_plot(explainer.expected_value[0], rounded_shap_values, rounded_features, matplotlib=True )

explainer = shap.KernelExplainer(model=nn_model.predict, data=background)

# Compute SHAP values - this might take some time for larger datasets.
shap_values = explainer.shap_values(background)

# Visualize the first prediction's explanation
instance_index = 5

# Round the SHAP values and the features of the instance you want to visualize
rounded_shap_values = np.round(shap_values[0][instance_index], 1)
rounded_features = np.round(X_train_wo.iloc[instance_index], 1)
rounded_shap_values, rounded_features
# Visualizing the SHAP values for the first prediction
shap.initjs() # Initialize JavaScript visualization in Jupyter notebooks if applicable
shap.force_plot(explainer.expected_value[0], rounded_shap_values, rounded_features, matplotlib=True )