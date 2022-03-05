import numpy as np
import pandas as pd
from datetime import datetime as dt
#import warnings
#warnings.filterwarnings('ignore')

exp_var = ['hate_crime', 'hate_crime_combined', 'hc_by_year', 'region_grouped', 'race_grouped', 'indexed_df']


#hate crime csv
hate_crime = pd.read_csv('hate_crime.csv', low_memory=False)

#political climate csv
political = pd.read_csv('political_climate.csv', low_memory=False)




population_groups = hate_crime[['POPULATION_GROUP_CODE', 'POPULATION_GROUP_DESC']].value_counts()
population_groups_df = pd.DataFrame(population_groups).sort_values('POPULATION_GROUP_CODE').reset_index()
population_groups_df = population_groups_df[['POPULATION_GROUP_CODE', 'POPULATION_GROUP_DESC']]


## PREPROCESSING

#dropping duplicate columns like state name and unnecessary columns like Agency Name
hate_crime = hate_crime.drop(['STATE_NAME', 'POPULATION_GROUP_DESC', 'PUB_AGENCY_UNIT', 
                              'ORI', 'PUB_AGENCY_NAME', 'AGENCY_TYPE_NAME', 
                              'TOTAL_INDIVIDUAL_VICTIMS','DIVISION_NAME', 'INCIDENT_ID', 
                              'MULTIPLE_OFFENSE', 'MULTIPLE_BIAS'], axis=1)

### checking null values
percent_missing = hate_crime.isnull().sum() *100/len(hate_crime)
missing_values_df = pd.DataFrame({'column_name': hate_crime.columns, 'percent_missing': percent_missing})
missing_values_df.sort_values('percent_missing', inplace = True)

#drop columns with more than 70% missing values
perc = 70.0
min_count = int(((100-perc)/100)*hate_crime.shape[0]+1)
hate_crime = hate_crime.dropna(axis=1, thresh=min_count)


### replace null values in OFFENDER RACE column
#view unique values
unique_race_cat = hate_crime['OFFENDER_RACE'].unique()

#replace nan with unknown label
hate_crime['OFFENDER_RACE'] = hate_crime['OFFENDER_RACE'].replace(np.nan, 'Unknown')
hate_crime['OFFENDER_RACE'].unique()


## TRANSFORMING DATATYPES
#convert to datetime
hate_crime["INCIDENT_DATE"] = pd.to_datetime(hate_crime["INCIDENT_DATE"])

#reduce the number of categories for VICTIM_TYPES by condensing labels
replacements = {'VICTIM_TYPES':{r'.*Law Enforcement Officer.*':'Law Enforcement Officer', 
                                r'.*Religious Organization.*': 'Religious Organization', 
                                r'.*Business.*': 'Business', 
                                r'.*Government.*': 'Government', 
                                r'.*Individual.*': 'Individual', 
                                r'.*Society/Public.*':'Society/Public'}, 
               'BIAS_DESC':{r'.*Anti-Black.*':'Anti-Black or African American', 
                             r'.*Anti-Jewish.*': 'Anti-Jewish', 
                             r'.*Anti-Gay.*': 'Anti-Gay (Male)',
                             r'.*Anti-Lesbian.*': 'Anti-Lesbian (Female)', 
                             r'.*Anti-Islamic.*': 'Anti-Islamic (Muslim)',
                             r'.*Anti-Hispanic.*': 'Anti-Hispanic or Latino',
                             r'.*Anti-Transgender.*': 'Anti-Transgender', 
                             r'.*Anti-Gender Non-Conforming.*': 'Anti-Gender Non-Conforming',
                             r'.*Anti-Asian.*': 'Anti-Asian',
                             r'.*Anti-Bisexual,*':'Anti-Bisexual',
                             r'.*Anti-American Indian.*': 'Anti-Native American',
                             r'.*Anti-Mental Disability.*': 'Anti-Mental Disability',
                             r'.*Anti-Physical Disability.*': 'Anti-Physical Disability',
                             r'.*Anti-Other Religion.*': 'Anti-Other Religion', 
                             r'.*Anti-Multiple Races, Group.*': 'Anti-Multiple Races, Group', 
                             r'.*Anti-Hindu.*': 'Anti-Hindu', 
                             r'.*Anti-Catholic.*': 'Anti-Catholic', 
                             r'.*Anti-Arab.*': 'Anti-Arab', 
                             r'.*Anti-Jehovah.*': 'Anti-Jehovahs Witness', 
                             r'.*Anti-White.*': 'Anti-White',
                             r'.*Anti-Multiple Religions.*': 'Anti-Multiple Religions',
                             r'.*Anti-Protestant.*': 'Anti-Protestant',
                             r'.*Anti-Native Hawaiian.*': 'Anti-Native Hawaiian or Other Pacific Islander',
                             r'.*Anti-Bisexual.*': 'Anti-Bisexual', 
                             r'.*Anti-Female.*': 'Anti-Female', 
                             r'.*Anti-Sikh.*': 'Anti-Sikh'}, 
               'LOCATION_NAME':{r'.*Highway/Road/Alley/Street/Sidewalk.*':'Highway/Road/Alley/Street/Sidewalk', 
                                 r'.*College.*': 'School-College/University', 
                                 r'.*Residence/Home.*': 'Residence/Home',
                                 r'.*Drug Store/Doctor.*': 'Drug Store/Doctor', 
                                 r'.*Commercial/Office Building.*': 'Commercial/Office Building',
                                 r'.*Restaurant.*': 'Restaurant', 
                                 r'.*Government/Public Building.*': 'Government/Public Building',
                                 r'.*Grocery/Supermarket.*': 'Grocery/Supermarket',
                                 r'.*Parking/Drop Lot/Garage.*': 'Parking/Drop Lot/Garage',
                                 r'.*Jail/Prison/Penitentiary/Corrections Facility.*': 'Jail/Prison/Penitentiary/Corrections Facility',  
                                 r'.*School-Elementary/Secondary.*': 'School-Elementary/Secondary', 
                                 r'.*Church/Synagogue/Temple/Mosque.*': 'Church/Synagogue/Temple/Mosque', 
                                 r'.*Amusement Park.*': 'Amusement Park',
                                 r'.*Bar/Nightclub.*': 'Bar/Nightclub',
                                 r'.*Air/Bus/Train Terminal.*': 'Air/Bus/Train Terminal',
                                 r'.*Department/Discount Store.*': 'Department/Discount Store',
                                 r'.*Auto Dealership New/Used.*': 'Auto Dealership New/Used'
                                }}
hate_crime.replace(replacements, regex=True, inplace=True)


### COMBINE hate_crime and political df


#create a new column to show the middle year between start and end years
political['Middle Year'] = political['Year Start'] + 1

#merge dataframes for all hate crimes that occured on a year in 'Start'
start = hate_crime.merge(political, how='inner', left_on='DATA_YEAR', right_on='Year Start')

#merge dataframes for all hate crimes that occured on a year in 'Middle'
middle = hate_crime.merge(political, how='inner', left_on='DATA_YEAR', right_on='Middle Year')

#concat Start and middle to form new combined dataframe
#no need to include end year because end year for one presidency overlaps with start year of next
hate_crime_combined = pd.concat([start, middle])
hate_crime_combined.drop('Middle Year', axis=1, inplace=True)


### AGGREGATE DATA

### YEAR GROUPED

hate_crime_combined['INCIDENT_COUNT'] = 1

year_sums = hate_crime_combined.groupby(['DATA_YEAR']).sum().reset_index()
hc_by_year = year_sums.drop(['Congress', 'Year Start', 'Year End'], axis=1)

hc_by_year['AVG_NO_OFFENDERS'] = (hc_by_year['TOTAL_OFFENDER_COUNT'] / hc_by_year['INCIDENT_COUNT'])
hc_by_year['AVG_NO_VICTIMS'] = (hc_by_year['VICTIM_COUNT'] / hc_by_year['INCIDENT_COUNT'])


year_modes = hate_crime_combined.groupby('DATA_YEAR')[['OFFENSE_NAME', 'VICTIM_TYPES', 'BIAS_DESC', 'Presidency']].agg(pd.Series.mode).reset_index()
hc_by_year.merge(year_modes)



## REGION GROUPED
region_grouped = hate_crime_combined.groupby(['REGION_NAME', 'DATA_YEAR'])['INCIDENT_COUNT'].sum().reset_index()


## RACE GROUPED
race_grouped = hate_crime_combined.groupby(['OFFENDER_RACE', 'DATA_YEAR'])['INCIDENT_COUNT'].sum().reset_index()


## TIME SERIES ANALYSIS
hate_crime['TOTAL_INCIDENTS']=1
ts_df = hate_crime[['INCIDENT_DATE', 'TOTAL_INCIDENTS']]

#create incident year and month column from incident date
ts_df['INCIDENT_MONTH'] = pd.to_datetime(ts_df['INCIDENT_DATE']).dt.to_period('M')

ts_df.drop('INCIDENT_DATE', axis=1, inplace=True)

#convert incident month to datetime with monthly frequency
ts_df['INCIDENT_MONTH'] = ts_df['INCIDENT_MONTH'].apply(lambda x: x.to_timestamp(freq='M'))

ts_df = ts_df.groupby(['INCIDENT_MONTH'])['TOTAL_INCIDENTS'].sum().reset_index()

indexed_df = ts_df.set_index('INCIDENT_MONTH')