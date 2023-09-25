# Databricks notebook source
# MAGIC %pip install delta-sharing

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to extract cleaned data from using delta share protocol

# COMMAND ----------

# DBTITLE 1,retrieving share credentials
import delta_sharing
dbutils.fs.cp('s3://hls-eng-data-public/delta_share/rearc_hls_data.share','/tmp/')
share_file_path = "/tmp/rearc_hls_data.share"
client = delta_sharing.SharingClient(f"/dbfs{share_file_path}")
shared_tables = client.list_all_tables()

# COMMAND ----------

# DBTITLE 1,setup deltashare
dataset_urls = {
  "bronze_income":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.bronze_income",
  "silver_poverty":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.poverty_county",
  "silver_education":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.education_county",
  "silver_health_stats":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.health_stats_county",
  "silver_race":f"{share_file_path}#rearc_databricks_hls_share.hls_sdoh.race_county",
  "silver_vaccinations":f"{share_file_path}#rearc_databricks_hls_share.hls_covid19_usa.vaccinations_county_utd",
}

# COMMAND ----------

# initiate schema
spark.sql("DROP SCHEMA IF EXISTS sdoh CASCADE")
spark.sql("CREATE SCHEMA sdoh")

# COMMAND ----------

# Add tables
for ds, url in dataset_urls.items():
  spark.sql(f"CREATE TABLE IF NOT EXISTS sdoh.{ds} USING deltaSharing LOCATION '{url}'")

# COMMAND ----------

# MAGIC %sql
# MAGIC use sdoh

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# DBTITLE 1,Education status
# MAGIC %sql
# MAGIC select * from silver_education limit 20;

# COMMAND ----------

# DBTITLE 1,Health status
# MAGIC %sql
# MAGIC select * from silver_health_stats limit 20;

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the correlation between different health stats. To do so, we use `Correlation` function from `pyspark.ml.stat` package to calculate pairwise pearson correlation coefficients among features of interest (such as smoking, obesity etc). This approach enables us to leverage distributed processing to accelerate computation.

# COMMAND ----------

# loading libraries
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import plotly.express as px

# COMMAND ----------

# create the dataframe of selected features
_df = sql("select SmokingPct, ObesityPct, HeartDiseasePct, CancerPct, NoHealthInsPct, AsthmaPct from silver_health_stats")

# COMMAND ----------

# convert columns of the dataframe to vectores
vecAssembler = VectorAssembler(outputCol="features")
vector_col = "corr_features"

# COMMAND ----------

assembler = VectorAssembler(inputCols=_df.columns, outputCol=vector_col)
df_vector = assembler.transform(_df).select(vector_col)

# COMMAND ----------

# calculating correlation matrixs
corr_matrix = Correlation.corr(df_vector, vector_col).select('pearson(corr_features)').collect()[0]['pearson(corr_features)'].toArray()

# COMMAND ----------

# DBTITLE 1,Pairwise correlation among different health stats
col_names = _df.columns
_pdf=pd.DataFrame(corr_matrix, columns=col_names, index=col_names)

# COMMAND ----------

px.imshow(_pdf, text_auto=True)

# COMMAND ----------

# MAGIC %md
# MAGIC From the matrix above we see a very significant correlation between rate of smoking and other risk factors such as obesity, heart disease and asthma. Perhaps a more rigorous analysis would require taking into account estimation errors due to the sizes of counties.

# COMMAND ----------

# DBTITLE 1,vaccinations
# MAGIC %sql
# MAGIC select * from silver_vaccinations limit 20

# COMMAND ----------

# DBTITLE 1,Vaccination rates for 12 and older across counties
from urllib.request import urlopen
import json
import pandas as pd
import plotly.express as px
import numpy as np

# COMMAND ----------

with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

# COMMAND ----------

_pdf = sql('select fips,avg(Series_Complete_12PlusPop_Pct) as vaccination_rate  from silver_vaccinations group by fips').toPandas()

# COMMAND ----------

fig = px.choropleth(_pdf, geojson=counties, locations='fips', color='vaccination_rate',
                           color_continuous_scale="Viridis",
                           scope="usa",
                           labels={'vaccination_rate':'vaccination_rate'}
                          )

# COMMAND ----------

fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building
# MAGIC
# MAGIC Now we proceed to create a dataset for downstream analysis using ML. The target value to predict is vaccination rate for people 12 years and older which is `Series_Complete_12PlusPop_Pct`

# COMMAND ----------

# MAGIC %md
# MAGIC #### Creating Training Data
# MAGIC Here, we also need population density data. In this case, we directly read the `csv` files and register the resulting dataset as a view.

# COMMAND ----------

# DBTITLE 1,adding population density
spark.read.csv('wasb://data@sdohworkshop.blob.core.windows.net/sdoh/Population_Density_By_County.csv', header=True, inferSchema=True).createOrReplaceTempView('population_density')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from population_density limit 20

# COMMAND ----------

# MAGIC %md Now that we have all the data, we create the dataset needed for our downstream analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC use sdoh;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW vaccine_data_pct
# MAGIC AS
# MAGIC (
# MAGIC   SELECT
# MAGIC     v.fips,
# MAGIC     Recip_County,
# MAGIC     Recip_State,
# MAGIC     ifnull(Series_Complete_12PlusPop_Pct, Series_Complete_Pop_Pct) AS Series_Complete_12PlusPop_Pct,
# MAGIC     pd.Density_per_square_mile_of_land_area AS population_density,
# MAGIC     r.County_Population,
# MAGIC     round((r.County_Population - r.White_Population) / r.County_Population, 3) * 100 AS Minority_Population_Pct,
# MAGIC     i.`2019` AS income,
# MAGIC     p.All_Ages_in_Poverty_Percent,
# MAGIC     round(e.25PlusHS / r.County_Population, 2) * 100 25PlusHSPct,
# MAGIC     round(e.25PlusAssociate / r.County_Population, 2) * 100 AS 25PlusAssociationPct,
# MAGIC     h.SmokingPct,
# MAGIC     h.ObesityPct,
# MAGIC     h.HeartDiseasePct,
# MAGIC     h.CancerPct,
# MAGIC     h.NoHealthInsPct,
# MAGIC     h.AsthmaPct
# MAGIC   FROM silver_race r JOIN sdoh.silver_vaccinations v ON(r.fips=v.fips)
# MAGIC   JOIN bronze_income i ON(i.geofips=v.fips)
# MAGIC   JOIN silver_poverty p ON(p.fips = v.fips)
# MAGIC   JOIN silver_education e ON(e.fips=v.fips)
# MAGIC   JOIN silver_health_stats h ON(h.locationid = v.fips)
# MAGIC   JOIN population_density pd ON(pd.GCT_STUBtarget_geo_id2 = v.fips)
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from vaccine_data_pct

# COMMAND ----------

# MAGIC %md
# MAGIC Now we create a pandas dataframe which will be used by ML framework

# COMMAND ----------

parsed_pd = spark.table("vaccine_data_pct").toPandas().dropna(subset=['Series_Complete_12PlusPop_Pct'])
parsed_pd.set_index('fips')

# COMMAND ----------

X = parsed_pd.drop(["Series_Complete_12PlusPop_Pct", "fips", "Recip_County", "Recip_State", "County_Population"], axis=1)
y = parsed_pd["Series_Complete_12PlusPop_Pct"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear Regression with XG Boost
# MAGIC Now, we use XG Boost to train a linear regression model and use MLFlow autolog to track model performance

# COMMAND ----------

# DBTITLE 1,importing libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np
import pandas as pd
import shap
import mlflow

# COMMAND ----------

mlflow.autolog()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

# COMMAND ----------

# DBTITLE 1,defining learning parameter for xgboost
max_depth = 1
learning_rate = .1
reg_alpha = .1

# COMMAND ----------

# MAGIC %md
# MAGIC we use `n_estimators=500` for to accelearte the training, in practice you may want to use higher values for more accurate results

# COMMAND ----------

# DBTITLE 1,defining the model
xgb_regressor = XGBRegressor(
    objective='reg:squarederror',
    max_depth=max_depth,learning_rate=learning_rate,
    reg_alpha=reg_alpha,
    n_estimators=500,
    importance_type='total_gain',
    random_state=0
)

# COMMAND ----------

# DBTITLE 1,fitting the training data
xgb_model = xgb_regressor.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=25
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Explainability and Feature Importance
# MAGIC we use SHAP values for better understanding our model's behaviour

# COMMAND ----------

explainer = shap.TreeExplainer(xgb_model)

# COMMAND ----------

shap_values = explainer.shap_values(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC Add FIPS Code to SHAP values

# COMMAND ----------

df = pd.DataFrame(shap_values, columns=X.columns)

# COMMAND ----------

df['fips'] = spark.sql("select fips from vaccine_data_pct order by fips").toPandas()

# COMMAND ----------

dfShaps=spark.createDataFrame(df)

# COMMAND ----------

dfShaps.createOrReplaceTempView("shap")

# COMMAND ----------

sql("select * from shap limit 20").display()

# COMMAND ----------

# MAGIC %md
# MAGIC Pivot the columns back to rows to make reporting easier and add to the database

# COMMAND ----------

usa_model_county_vaccine_shap_df = sql("""
select fips, stack(12,'Minority_Population_Pct',Minoirity_Population_Pct,'income', income, '25PlusHSPct', 25PlusHSPct,
'All_Ages_in_Poverty_Percent',All_Ages_in_Poverty_Percent,'population_density', population_density, '25PlusAssociatePct', 25PlusAssociatePct, 
'SmokingPct',SmokingPct, 
'ObesityPct', ObesityPct, 
'HeartDiseasePct', HeartDiseasePct, 
'CancerPct', CancerPct, 
'NoHealthInsPct', NoHealthInsPct,
'AsthmaPct', AsthmaPct
)  
as (factor, value)
from shap
""").limit(50)
display(usa_model_county_vaccine_shap_df)

# COMMAND ----------

# Top SDH effecting vaccination rates 
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
_pdf=pd.DataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:6],columns=["Mean(SHAP)", "Column"])
px.bar(_pdf,x='Column',y='Mean(SHAP)')
