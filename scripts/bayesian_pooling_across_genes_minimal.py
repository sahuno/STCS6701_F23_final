#shared pooling across genes
import numpy as np
import pandas as pd
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam
import arviz as az
import pickle
import jax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.preprocessing import LabelEncoder



#load the data
with open('train_methyl_RNA.pkl', 'rb') as fp:
    train = pickle.load(fp)

# pooling across genes
def Normal_model_gene_pooling(geneID_Code, X_treatment, X_methylation, Y_RNA=None):
    μ_βg_μ = numpyro.sample("μ_βg_μ", dist.Normal(0.0, 10.0))
    σ_βg_μ = numpyro.sample("σ_βg_μ", dist.HalfNormal(3.0))
    μ_βg_M = numpyro.sample("μ_βg_M", dist.Normal(0.0, 10.0))
    σ_βg_M = numpyro.sample("σ_βg_M", dist.HalfNormal(3.0))
    n_genes = len(np.unique(geneID_Code))
    with numpyro.plate(str("Genes N="+str(n_genes)), n_genes):
        βg_μ = numpyro.sample("βg_μ", dist.Normal(μ_βg_μ, σ_βg_μ))
        βg_M = numpyro.sample("βg_M", dist.Normal(μ_βg_M, σ_βg_M))
    σ = numpyro.sample("σ", dist.LogNormal(0.0, 1.0))
    #if geneID_Code = 3; βg_M[geneID_Code] is getting `βg_M` value for geneID_Code=3; note that geneID_Code=3 can appear multiple times in my data.
    RNA_est = βg_μ[geneID_Code] + βg_μ[geneID_Code] * X_treatment + βg_M[geneID_Code] * X_methylation
    RNA_est = jnp.array(RNA_est)
    with numpyro.plate("data", len(geneID_Code)):
       obs = numpyro.sample("Y_g", dist.Normal(RNA_est, σ), obs=Y_RNA)
    return RNA_est

# pooling across genes
# def Normal_model_gene_pooling_with_patients(geneID_Code, subjectCode, fraction_modified_entropy_in_log2, RNA_exp=None):
#     # extend model to include patient-dependent latent variables
#     # ...
#     μ_βg_μ = numpyro.sample("μ_βg_μ", dist.Normal(0.0, 10.0))
#     σ_βg_μ = numpyro.sample("σ_βg_μ", dist.HalfNormal(3.0))
#     μ_βg_M = numpyro.sample("μ_βg_M", dist.Normal(0.0, 10.0))
#     σ_βg_M = numpyro.sample("σ_βg_M", dist.HalfNormal(3.0))
#     n_genes = len(np.unique(geneID_Code))
#     n_subjectCode = len(np.unique(subjectCode))
#     σ = numpyro.sample("σ", dist.LogNormal(0.0, 1.0))
#     σ_subj = numpyro.sample("σ_subj", dist.LogNormal(0.0, 1.0))
#     with numpyro.plate(str("subjects N="+str(n_genes)), n_subjectCode):
#         βg_μ_j = numpyro.sample("βg_μ_j", dist.Normal(0.0, 10))
#         βg_M_j = numpyro.sample("βg_M_j", dist.Normal(μ_βg_M, σ_βg_M))
#     RNA_est_subj = βg_μ_j[subjectCode] + βg_M_j[subjectCode] * fraction_modified_entropy_in_log2
#     with numpyro.plate(str("Genes N="+str(n_genes)), n_genes):
#         #βg_μ = numpyro.sample("βg_μ", dist.Normal(μ_βg_μ, σ_βg_μ))
#         βg_M = numpyro.sample("βg_M", dist.Normal(RNA_est_subj, σ_subj))
#     RNA_est = βg_μ[geneID_Code] + βg_M[geneID_Code] * fraction_modified_entropy_in_log2
#     RNA_est = jnp.array(RNA_est)
#     with numpyro.plate("data", len(geneID_Code)):
#        numpyro.sample("Y_g", dist.Normal(RNA_est, σ), obs=RNA_exp)


#####################################
#use numpy instead of jax
# geneID_Code = train["geneID_Code"].values
# fraction_modified_entropy_in_log2 = train["fraction_modified_entropy_in_log2"].values
# RNA_exp = train["log_RPKM"].values
#####################################
#####################################



filtered_data = train[train["subject"].isin(["S-0-1", "S-0-2", "S-0-3", "S-A-1", "S-A-2","S-A-3"])]
filtered_dmso_azct_extra_cond_counts = filtered_data['geneID'].value_counts() 
filtered_dmso_azct_extra_cond_counts = filtered_dmso_azct_extra_cond_counts[filtered_dmso_azct_extra_cond_counts > 5]
print(filtered_dmso_azct_extra_cond_counts)
geneIds_filtered = filtered_dmso_azct_extra_cond_counts.index.tolist()  # Convert to a list if needed
# filtered_data

def categorize(subject):
  """
  This function assigns labels (0 or 1) based on the subject prefix.
  """
  if subject.startswith('S-0'):
    return 0
  else:
    return 1

#get only genes that is represented by everny sample
filtered_data_min = filtered_data[filtered_data["geneID"].isin(geneIds_filtered)].copy()
filtered_data_min['conditions'] = filtered_data_min['subject'].apply(categorize)
# filtered_data_min.shape

#creat trestment labels
treatment_status_encoder = LabelEncoder()
filtered_data_min["treatment_code"] = treatment_status_encoder.fit_transform(filtered_data_min["conditions"])


#algortithm 
#1. group-data by geneID, 
#2. schuffle each group of grouped-data, 
#2. take any 2 out of 3 subjects in each treatment training leaving 1/3 in for each trt as testset
filtered_data_min_group_schuffled = filtered_data_min[filtered_data_min["geneID"].isin(["ENSMUSG00000025333.10", "ENSMUSG00000025903.14", "ENSMUSG00000063663.11"])]
filtered_data_min_group_schuffled = filtered_data_min_group_schuffled[["geneID", "treatment_code","subjectCode","fraction_modified_entropy_in_log2", "log_RPKM"]]
####make small data to test out splitting techniques
filtered_data1G = filtered_data_min[filtered_data_min["geneID"].isin(["ENSMUSG00000025333.10"])]
filtered_data1G = filtered_data1G[["geneID", "treatment_code","subjectCode","fraction_modified_entropy_in_log2", "log_RPKM"]]


patient_encoder = LabelEncoder()
filtered_data_min.loc[:, "subjectCode"] = patient_encoder.fit_transform(filtered_data_min["subject"].values)

#make gene codes for 
gene_encoder = LabelEncoder()
filtered_data_min["geneID_Code"] = gene_encoder.fit_transform(filtered_data_min["geneID"].values)
filtered_data_min.loc[:, "geneID_Code"] = gene_encoder.fit_transform(filtered_data_min["geneID"].values)



#function to schuffle data and split into train and validation set
def splitGroups(groups, size=0.2):
    try:
        group_train_set, group_test_set = train_test_split(groups, test_size=size, random_state=0, stratify=groups[["treatment_code"]])
        output = {'trainSet': group_train_set, 'testSet': group_test_set}
        return output
    except Exception as e:
        print(f"Error encountered on iteration")

#define function  to split
def unnest_train_test(dict, newColName):
    train_list = {}
    test_list = {}
    for i,v in dict.items():
        train_list[i] = v['trainSet']
        test_list[i] = v['testSet']
    train_splitted_df = pd.concat(train_list.values(), ignore_index=False,  keys= train_list.keys(), names=[newColName]).reset_index(0)
    test_splitted_df = pd.concat(test_list.values(), ignore_index=False, keys= test_list.keys(), names=[newColName]).reset_index(0)
    return train_splitted_df, test_splitted_df


schuffled_grouped_split = filtered_data_min.groupby('geneID').apply(lambda x: splitGroups(x.sample(frac=1)), include_groups=False)
train_filtered, test_filtered  = unnest_train_test(dict=schuffled_grouped_split, newColName = 'geneID')

data2Save = {"train_filtered":train_filtered, "test_filtered":test_filtered}
with open('trainTestSets_methyl_RNA_DMSOvrsAZCT_only.pkl', 'wb') as file: 
    pickle.dump(data2Save, file) 


# with open('trainTestSets_methyl_RNA_DMSOvrsAZCT_only.pkl', 'rb') as fp:
#     data2Save = pickle.load(fp)
# train_filtered = data2Save["train_filtered"]
# test_filtered = data2Save["test_filtered"]






#create jax datasets
def data2jnp(pdDataframe):
    #copy to jax! for fast computaion
    geneID_Code = jnp.array(pdDataframe["geneID_Code"].values)
    X_treatment = jnp.array(pdDataframe["treatment_code"].values)
    X_methylation = jnp.array(pdDataframe["fraction_modified_entropy_in_log2"].values)
    Y_RNA = jnp.array(pdDataframe["log_RPKM"].values)
    return geneID_Code, X_treatment, X_methylation, Y_RNA

geneID_Code_train, X_treatment_train, X_methylation_train, Y_RNA_train = data2jnp(pdDataframe=train_filtered)
geneID_Code_test, X_treatment_test, X_methylation_test, Y_RNA_test = data2jnp(pdDataframe=test_filtered)
geneID_Code_train.shape, X_treatment_train.shape, X_methylation_train.shape, Y_RNA_train.shape
X_treatment_test.shape

# def render_models(modelName = None, save_filename=None):
    # numpyro.render_model(model=modelName, model_args=(geneID_Code=geneID_Code_train, X_treatment=X_treatment_train, X_methylation = X_methylation_train, Y_RNA = Y_RNA_train,), render_params=True, render_distributions=True, filename=save_filename)

# numpyro.render_model(model=Normal_model_gene_fitGenes, model_args=(geneID_Code, fraction_modified_entropy_in_log2, RNA_exp,), render_params=True, render_distributions=True, filename="/figures/model_methyl_rna_genes_indepdently.png")
numpyro.render_model(model=Normal_model_gene_pooling, model_args=(geneID_Code_train, X_treatment_train, X_methylation_train, Y_RNA_train,), render_params=True, render_distributions=True, filename="/figures/model_methyl_rna_pooled_across_genes_minimal.png")
# numpyro.render_model(model=Normal_model_gene, model_args=(geneID_Code, fraction_modified_entropy_in_log2, RNA_exp,), render_params=True, render_distributions=True, filename="figures/model_methyl_rna_pooled_across_genes_v2.png")


# , num_chains=1
def mcmc_infer(model=None):
    print("running NUTS mcmc")
    nuts_kernel = NUTS(model=model)
    # jax.device_get(nuts_kernel)
    mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=2000, num_chains=1)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, geneID_Code_train, X_treatment_train, X_methylation_train, Y_RNA_train)
    posterior_samples = mcmc.get_samples()
    return mcmc, posterior_samples


#jax.device_get(mcmc)
#run mcmc
mcmc, posterior_samples_Normal = mcmc_infer(model = Normal_model_gene_pooling)
#mcmc.print_summary()
posterior_samples_Normal['βg_M'].shape #oh this is actualy equal number of unique genes that i passed in


posterior_samples_Normal.keys()
data_pooled_across_genes = az.from_numpyro(mcmc)
figaz = az.plot_trace(data_pooled_across_genes, compact=True, figsize=(15, 25))
fig = figaz.ravel()[0].figure
fig.savefig("figures/learntParams_pooled_across_genes_minimal_2Train1TestV2.png")

# az.summary(cookies_trace)


##TODO; dress this function learning from tutorials
#Predictive
##posterior perdictive
def posterior_pred(modelObject, sites2Return, posteriorSamplesObj):
    predictive = Predictive(model = modelObject, posterior_samples=posteriorSamplesObj, return_sites=sites2Return)
    samples_predictive = predictive(random.PRNGKey(0), geneID_Code_train, X_treatment_train, X_methylation_train, None)
    return samples_predictive
    

#posterior predictive with unsen data
predictive = Predictive(model = Normal_model_gene_pooling, posterior_samples=posterior_samples_Normal, return_sites=('σ', 'Y_g', '_RETURN'))
samples_predictive = predictive(random.PRNGKey(0), geneID_Code_test, X_treatment_test, X_methylation_test, None)
samples_predictive.keys()

# from numpyro import handlers
# def fitted_means(rng_key, params, model, *args, **kwargs):
#     model = handlers.substitute(handlers.seed(model, rng_key), params)
#     model_trace = handlers.trace(model).get_trace(*args, **kwargs)
#     obs_node = model_trace['obs']
#     means_node = model_trace['_RETURN']
#     return means_node, obs_node



# samples_predictive_train = predictive(random.PRNGKey(0), geneID_Code_train, X_treatment_train, X_methylation_train, None)
# samples_predictive_train.keys()

#approach from; https://pyro.ai/examples/bayesian_regression_ii.html
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats


samples_predictive_summary = summary(samples_predictive).items()
# for site, values in summary(samples_predictive).items():
#     print("Site: {}".format(site))
#     print(values, "\n")


fullDf = pd.concat([train_filtered, test_filtered])
fullDf = fullDf[["subjectCode", "subject", "geneID", "geneID_Code", "fraction_modified_entropy_in_log2", "log_RPKM", "treatment_code"]]
#take samples from posterior
test_minimal =  test_filtered[["subjectCode", "subject", "geneID", "geneID_Code", "fraction_modified_entropy_in_log2", "log_RPKM", "treatment_code"]].copy()
test_minimal["Y_g_pred"] = np.array(samples_predictive["Y_g"].T.mean(axis=1))
test_minimal["Y_g_sigma_pred"] = np.array(samples_predictive["Y_g"].T.std(axis=1))

# train_minimal["Y_g_pred"] 
test_minimal["Y_g_pred_LowerCI"]  = test_minimal["Y_g_pred"] - test_minimal["Y_g_sigma_pred"]
test_minimal["Y_g_pred_HigherCI"] = test_minimal["Y_g_pred"] + test_minimal["Y_g_sigma_pred"]

rmse = ((test_minimal["log_RPKM"] - test_minimal["Y_g_pred"]) ** 2).mean() ** (1 / 2)
print(f"RMSE: {rmse:.1f}")


#plot residuals vrs y true
test_residuals = (test_minimal["log_RPKM"] - test_minimal["Y_g_pred"]) 
plt.scatter(test_residuals.values, test_minimal["log_RPKM"].values, alpha = 0.1)
plt.xlabel("Residuals [Y_true - Y_pred]")
plt.ylabel("True log RPKM")
plt.title("Residuals vs True log RPKM on hold-out data")
plt.savefig('residuals_vrs_true_logRPKM.png', format='png', dpi=300, bbox_inches='tight')
plt.close('all')
#########################
######### save models 
# import dill
# output_dict = {}
# output_dict['model']=Normal_model_gene_pooling
# output_dict['mcmc']=mcmc
# output_dict['samples_predictive']=samples_predictive
# with open('mcmc_model.pkl', 'wb') as handle:
#     dill.dump(output_dict, handle)

# posterior_samples.keys() 
# np.min(test_residuals)

modelEvalsData = {"test_residuals":test_residuals, "rmse":rmse, "test_minimal":test_minimal, "fullDf":fullDf, "mcmc":mcmc, "posterior_samples_Normal":posterior_samples_Normal}
with open('modelEvalObjects_DMSOvrsAZCT_only.pkl', 'wb') as file: 
    pickle.dump(modelEvalsData, file) 

# with open('samples_predictive_summary_DMSOvrsAZCT_only.pkl', 'wb') as file: 
#     pickle.dump(list(samples_predictive_summary.items()), file) 

# samples_predictive_summary.keys()
# modelEvalsData["test_residuals"] = test_residuals

####plot per gene 
# def chart_subject_with_predictions(patient_id, ax):
#     data = test_minimal[test_minimal["subject"] == patient_id]
#     x = data["fraction_modified_entropy_in_log2"]
#     ax.set_title(patient_id)
#     ax.plot(x, data["log_RPKM"], "o")
#     ax.plot(x, data["Y_g_pred"])
#     ax = sns.regplot(x=x, y=data["log_RPKM"], ax=ax, ci=None, line_kws={"color": "red"})
#     ax.fill_between(x, data["Y_g_pred_LowerCI"], data["Y_g_pred_HigherCI"], alpha=0.5, color="#ffcd3c")
#     ax.set_ylabel("log_RPKM")

# df2plot = fullDf[fullDf["geneID"] == "ENSMUSG00000000001.4"][["geneID", "treatment_code","subjectCode","fraction_modified_entropy_in_log2", "log_RPKM"]]
# groups = df2plot["treatment_code"].apply(str)  # Extract unique groups
# palette = sns.color_palette("bright", n_colors=2)  # Generate colors
# color_map = dict(zip(groups, palette)) 
# fig1, axes = plt.subplots(1, 2, figsize=(15, 5))
# axes[0].plot(df2plot["fraction_modified_entropy_in_log2"].values, df2plot["log_RPKM"].values, "o", color=df2plot["treatment_code"].map(color_map))
# fig1.savefig('test_colorbytreatment.png', format='png', dpi=300, bbox_inches='tight')
# plt.close(fig1)  # Close the plot figure to free up memory


def chart_genes_with_predictions(geneID, ax):
    geneInFullData = fullDf[fullDf["geneID"] == geneID]
    data = test_minimal[test_minimal["geneID"] == geneID]
    x = data["fraction_modified_entropy_in_log2"]
    xInFullData = geneInFullData["fraction_modified_entropy_in_log2"]
    palette = sns.color_palette("bright", n_colors=geneInFullData['treatment_code'].nunique())
    color_map = dict(zip(geneInFullData['treatment_code'].unique(), palette))
    scatter_colors = geneInFullData['treatment_code'].map(color_map)
    ax.set_title(geneID)
    # ax.scatter(xInFullData, geneInFullData["log_RPKM"], marker="o", c=geneInFullData["treatment_code"].map(color_map), s=50)
    ax.plot(x, data["Y_g_pred"])
    #scatter_kws={'color': scatter_colors}
    ax = sns.regplot(x=geneInFullData["fraction_modified_entropy_in_log2"], y=geneInFullData["log_RPKM"], ax=ax, ci=None, line_kws={"color": "darkorange"})
    ax.fill_between(x, data["Y_g_pred_LowerCI"], data["Y_g_pred_HigherCI"], alpha=0.1, color="blue")
    ax.set_ylabel("log_RPKM")

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart_genes_with_predictions("ENSMUSG00000000001.4", axes[0])
chart_genes_with_predictions("ENSMUSG00000000028.15", axes[1])
chart_genes_with_predictions("ENSMUSG00000118607.1", axes[2])
f.savefig("posterior_minimal_genes_methyl_rna_minimal_2Train1TestV2.png") 

index_of_min = test_residuals.idxmin()
index_of_max = test_residuals.idxmax()

f, axes = plt.subplots(1, 4, figsize=(15, 5))
chart_genes_with_predictions(test_minimal.loc[index_of_min]["geneID"], axes[0])
chart_genes_with_predictions(test_minimal.loc[index_of_max]["geneID"], axes[1])
chart_genes_with_predictions("ENSMUSG00000025027.18", axes[2])
chart_genes_with_predictions("ENSMUSG00000017724.14", axes[3])
f.suptitle("min & max residuals; lowest & highest SD log(rpkm)")
f.savefig("posterior_minMaxResiduals_lowest_highestSD_genes_methyl_rna_2Train1TestV2.png") 
plt.close('all')







# f, axes = plt.subplots(1, 3, figsize=(15, 5))
# chart_subject_with_predictions("S-0-1", axes[0])
# chart_subject_with_predictions("S-0-2", axes[1])
# chart_subject_with_predictions("S-0-3", axes[2])
# f.savefig("posterior_methyl_rna.png") 
subIDs = sorted(test_minimal["subject"].unique().tolist())
f, axes = plt.subplots(4, 4, figsize=(15, 12)) 
for i, subID in enumerate(subIDs):
    row = i // 4  # Calculate the row index
    col = i % 4   # Calculate the column index
    ax = axes[row, col]  # Get the axis for the current subplot
    chart_subject_with_predictions(patient_id=subID, ax=ax)

f.supxlabel('fraction_modified_entropy_in_log2')
f.supylabel('log_RPKM')
f.savefig("posterior_methyl_rna_minimal_2Train1Test.png") 

print("done goodbye")


# import pickle
# PIK = "session_bayesian_LinearReg_pooling_across_genes_with_validation_set.dat"
# obj_session = globals()
# with open(PIK, "wb") as f:
#     for value in obj_session:
#         pickle.dump(value, f)

# def saveall(filename):
#     with open(filename, "wb") as f:
#         while True:
#             try:
#                 yield pickle.dump(f)
#             except EOFError:
#                 break

# globals().keys()
# globals()[10]
# items = saveall("session_bayesian_LinearReg_pooling_across_genes_with_validation_set.dat")
# items.close()


# with open('session_bayesian_LinearReg_pooling_across_genes_with_validation_set.dat', 'rb') as fp:
#     globals_svaes = pickle.load(fp)