#standardSQL
# Lookup what IRS code values mean
SELECT
  irs_990_field,
  code,
  description
FROM
  `qwiklabs-gcp-01-12366c421db4.my_dataset_000001.irs990_code_lookup` # change 
WHERE
  irs_990_field IN ('elf','subcd');
  
#standardSQL
SELECT DISTINCT 
  fullVisitorId, 
  date 
FROM `data-to-insights.ecommerce.all_sessions_raw`
WHERE date = '20180708'
LIMIT 5;

#standardSQL
# Partition
 CREATE OR REPLACE TABLE ecommerce.partition_by_day
 PARTITION BY date_formatted
 OPTIONS(
   description="a table partitioned by date"
 ) AS
 SELECT DISTINCT 
 PARSE_DATE("%Y%m%d", date) AS date_formatted,
 fullvisitorId
 FROM `data-to-insights.ecommerce.all_sessions_raw`;

'''

    Table name: ecommerce.days_with_rain
    Use the date field as your PARTITION BY
    For OPTIONS, specify partition_expiration_days = 90
    Add the table description = "weather stations with precipitation, partitioned by day"

'''


#standardSQL
 CREATE OR REPLACE TABLE ecommerce.days_with_rain
 PARTITION BY date
 OPTIONS (
   partition_expiration_days=90,
   description="weather stations with precipitation, partitioned by day"
 ) AS
 SELECT
   DATE(CAST(year AS INT64), CAST(mo AS INT64), CAST(da AS INT64)) AS date,
   (SELECT ANY_VALUE(name) FROM `bigquery-public-data.noaa_gsod.stations` AS stations
    WHERE stations.usaf = stn) AS station_name,  -- Stations may have multiple names
   prcp
 FROM `bigquery-public-data.noaa_gsod.gsod*` AS weather
 WHERE prcp < 99.9  -- Filter unknown values
   AND prcp > 0      -- Filter stations/days with no precipitation
   AND _TABLE_SUFFIX = CAST( EXTRACT(YEAR FROM CURRENT_DATE()) AS STRING);
   
#standardSQL
# avg monthly precipitation
SELECT 
  AVG(prcp) AS average,
  station_name,
  date,
  DATE_DIFF(CURRENT_DATE(), date, DAY) AS partition_age,
EXTRACT(MONTH FROM date) AS month
FROM ecommerce.days_with_rain
WHERE station_name = 'WAKAYAMA' #Japan
GROUP BY station_name, date, month, partition_age
ORDER BY date;

# https://cloud.google.com/bigquery/docs/partitioned-tables
# Google BigQuery