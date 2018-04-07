library(forecast) 
library(fpp) 
install.packages("caret")
library(caret) 
install.packages("neuralnet")
library(neuralnet) 
install.packages("randomForest")
library(randomForest) 
install.packages("psych")
library(psych)
install.packages("VIM")
library(VIM) 
install.packages("mice")
library(mice) 
install.packages("ResourceSelection")
library(ResourceSelection) 
install.packages("corrplot")
library(corrplot) 
install.packages("party")
library(party)
install.packages("magrittr")
library(magrittr)
library('dplyr')
library(stats)
library(ggplot2)
library(scales)



train_data_with_total_cases=read.csv("C:/Users/sachi/Desktop/Data Science/Predictive Analytics/Dengue Competition/data new dengue/dengue_features_train_combined.csv")
nrow(train_data_with_total_cases)
ncol(train_data_with_total_cases)

testdata=read.csv("C:/Users/sachi/Desktop/Data Science/Predictive Analytics/Dengue Competition/data new dengue/dengue_features_test.csv")
ncol(testdata) 
nrow(testdata)
testdata
aggr(train_data_with_total_cases,prop = T ,numbers=T, plot = F)


#head(mydata[,-25])
#head(mytestdata)

#full = bind_rows(mydata[,-25],mytestdata)
all_variables = c('ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm',
             'reanalysis_air_temp_k','reanalysis_avg_temp_k',
             'reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k',
             'reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2',
             'reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm',
             'reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k',
             'station_avg_temp_c','station_diur_temp_rng_c',
             'station_max_temp_c','station_min_temp_c','station_precip_mm')


features_temp_related =c('reanalysis_min_air_temp_k',
             'reanalysis_specific_humidity_g_per_kg',
             'reanalysis_dew_point_temp_k',
             'station_avg_temp_c',
             'station_min_temp_c')



#removing all NA values with Non NA values
train_data_with_total_cases[all_variables] %<>% na.locf(fromLast = TRUE)

#all NA values removed with prior NA values

aggr(train_data_with_total_cases, prop = T, numbers =T, plot = F)
#NA values are fixed 


# now diving data into two datasets i.e one is of SJ and other is of IQ
train_sj=train_data_with_total_cases[1:936,] 
nrow(train_sj)
ncol(train_sj)
str(train_sj)
train_iq=train_data_with_total_cases[937:1456,] 
nrow(train_iq)
train_iq
#Removing all NA values with prior non-NA values
aggr(testdata[all_variables], prop = T, numbers = T, plot = T)

testdata[all_variables] %<>% na.locf(fromLast = TRUE)
(testdata)
#checking test data for NA values
aggr(testdata[all_variables], prop = T, numbers = T, plot = T)

#Dividing the test data into SJ and IQ 
sj_testdata = testdata[1:260,]
iq_testdata = testdata[261:416,]

#fingding missing values in test data and replacing them with Non-NA values
aggr(sj_testdata, prop = T, numbers= T, plot = F)

aggr(iq_testdata, prop = T, numbers = T, plot = F)
#removing values with Non-NA values

#finding correlation for the train data 
#before that check the missing values in train_iq and train_sj
aggr(train_iq, prop=T,numbers=T)
aggr(train_sj, prop=T,numbers=T)
#no missing value find
#correlation checking 
correlation_iq=cor(train_iq[,5:25])
corrplot(correlation_iq, method="square", type="lower", tl.col = "black") 

correlation_sj=cor(train_sj[,5:25]) 
corrplot(correlation_sj, method="circle", type="lower", tl.col = "black") 

#now analysing data using some time series decomposition and plotting time series
par(mfrow = c(1,2))
sj_time_series <- ts(train_sj$total_cases, frequency = 52, start = c(1990,30,04))

sj_time_series
class(sj_time_series)
iq_time_series <- ts(train_iq$total_cases, frequency = 52, start = c(2000,07,01))


plot(sj_time_series, ylab = "No. of cases",xlab = "year",
     main = "Time series for total cases in San Jaun ",ylim = c(0,500),col = "green")

plot(iq_time_series,ylab = "No. of cases",xlab = "year",
     main = "Time series for total cases in iquitos",ylim = c(0,500),col = "red")
#now decompsing using stl function to analyse it fully 
#stl and checking SSE, applying holt's winter 
stl_sj_time_series <- stl(sj_time_series,s.window = "periodic")
summary(stl_sj_time_series)
plot(stl_sj_time_series,main = " Using stl decompose for San Juan time series")




stl_iq_time_series <- stl(iq_time_series,s.window = "periodic")
summary(stl_iq_time_series)
plot(stl_iq_time_series, main = " Using stl decompose for Iquitos time series" )

# plotting ACF and PACF for the time series above , that we have created 
par(mfrow = c(1,2))
Acf(sj_time_series, main = "Autocorrelation of San Juna time series")
Pacf(sj_time_series,main = "Partial Autocorrelation of San Juna time series") # time series is not stationary , as it has ACF and PACF values 

par(mfrow = c(1,2))
Acf(iq_time_series, main = "Autocorrelation of Iquitos time series")
Pacf(iq_time_series,main = "Partial Autocorrelation of Iquitos time series")
#using test to see whether diffrencing is required in series or not
adf.test(sj_time_series, alternative = "stationary")
adf.test(iq_time_series,alternative = "stationary")
# p= 0.01 gives data is not stationary , so diffrencing is required
#now using ndiffs function to find out diffrence and seasonal difference times
ndiffs(sj_time_series) # 1 difference is required
ndiffs(iq_time_series)# 1 difference is required

nsdiffs(sj_time_series)# 0 seasonal difference is required
nsdiffs(iq_time_series) # 0 seasonal difference is required 

# using tsdisplay to see the time series 
class(sj_time_series)
#tsdisplay in future 
#-----------------------------Models-----------------------------#
#-----------AUTOARIMA--------------------------------------------#




auto_arima_sj <- auto.arima(sj_time_series, xreg = train_sj[features_temp_related])
summary(auto_arima_sj)
forecast_autoarima_sj <- forecast(auto_arima_sj , h = 260,xreg = sj_testdata[features_temp_related])
summary(forecast_autoarima_sj)
plot(forecast_autoarima_sj, main = " San Juan Forecast with ARIMA(1,1,1)", col = "green")


auto_arima_iq <- auto.arima(iq_time_series, xreg = train_iq[features_temp_related])
summary(auto_arima_iq)
forecast_autoarima_iq <- forecast(auto_arima_iq , h = 156,xreg = iq_testdata[features_temp_related])
summary(forecast_autoarima_iq)
plot(forecast_autoarima_iq, main = " Iquitos Forecast with ARIMA(1,0,4)", col = "red")



####

par(mfrow = c(1,2))
plot(forecast_autoarima_sj, main = " San Juan Forecast with ARIMA(1,1,1)", col = "green")
plot(forecast_autoarima_iq, main = " Iquitos Forecast with ARIMA(1,0,4)", col = "red")

#----------------------------applying  seasonal arima with regressors  ----------------#
par(mfrow = c(1,2))
fit_sj_time_Series <- Arima(sj_time_series, order=c(1,1,1), seasonal=c(0,0,0), xreg =  train_sj[features_temp_related] ) 
forecast_Arima_sj=forecast(fit_sj_time_Series, h = 260, xreg = sj_testdata[features_temp_related])
summary(forecast_Arima_sj)
plot(forecast_Arima_sj, col = "green", main = "San Juan Forecast with ARIMA(1,1,1)")
Box.test(residuals(forecast_Arima_sj),fitdf = 7, lag = 151,type="Ljung")  
plot(fit_sj_time_Series$fitted, col = "green")
lines(sj_time_series, col = "red")

fit_iq_time_Series <- Arima(iq_time_series, order=c(0,1,2), seasonal=c(0,0,1), xreg =  train_iq[features_temp_related] ) 
forecast_Arima_iq=forecast(fit_iq_time_Series, h = 156, xreg = iq_testdata[features_temp_related])
summary(forecast_Arima_iq)
plot(forecast_Arima_iq, col = "red",main = "Iquitos Forecast with ARIMA(0,1,2)[52]")
Box.test(residuals(forecast_Arima_sj),fitdf = 7, lag = 151,type="Box-Pierce")  
plot(fit_sj_time_Series$fitted, col = "green")
lines(sj_time_series, col = "red")


par(mfrow = c(2,1))
plot(forecast_Arima_sj$fitted,residuals(forecast_Arima_sj))
plot(forecast_Arima_iq$fitted,residuals(forecast_Arima_iq))

tsdisplay()
#Analysis
#tsdisplay for Sj and iq
tsdisplay(residuals(forecast_Arima_sj))
tsdisplay(residuals(forecast_Arima_iq))
#histogram
par(mfrow = c(2,1))
hist(residuals(forecast_Arima_sj))
hist(residuals(forecast_Arima_iq))




Box.test(residuals(forecast_Arima_iq),fitdf = 8, lag = 151,type="Box-Pierce")  
plot(fit_iq_time_Series$fitted, col = "green")
lines(iq_time_series, col = "red")

sj.final1 <- data.frame(sj_testdata[,1:3],total_cases = round(forecast_Arima_sj$mean))
iq.final1 <- data.frame(iq_testdata[,1:3],total_cases = round(forecast_Arima_iq$mean))
final1 <- bind_rows(sj.final1,iq.final1)
str(final1)
nrow(final1)
write.csv(final1,"arima_model_final.csv",row.names = F)
getwd()
head(final1)
#--------------------------------seasonal arima withour xreg------------------------------------------#


fit_sea_sj <- Arima(sj_time_series, order=c(1,1,1), seasonal=c(0,0,0)) 
fit_Arima_sea_sj=forecast(fit_sea_sj, h = 260)
summary(fit_Arima_sea_sj)
plot(fit_Arima_sea_sj)
tsdisplay(residuals(fit_sea_sj))

Box.test(residuals(fit_Arima_sea_sj),fitdf=2, lag=151, type="Ljung")  

fit_sea_iq <- Arima(iq_time_series, order=c(0,1,2), seasonal=c(0,0,1)) 
fit_Arima_sea_iq=forecast(fit_sea_iq, h = 156)
summary(fit_Arima_sea_iq)
plot(fit_Arima_sea_iq)
tsdisplay(residuals(fit_sea_iq))

Box.test(residuals(fit_Arima_sea_iq), fitdf=3, lag=151, type="Ljung")
# histogram
par(mfrow = c(2,1))
hist(residuals(fit_sea_sj))
hist(residuals(fit_sea_iq))

sj.final2 <- data.frame(sj_testdata[,1:3],total_cases = round(fit_Arima_sea_sj$mean))
iq.final2 <- data.frame(iq_testdata[,1:3],total_cases = round(fit_Arima_sea_iq$mean))
final2 <- bind_rows(sj.final2,iq.final2)

write.csv(final2,"arima_model_final2.csv",row.names = F)
getwd()

nrow(final2)
#-------------------------Nueral Network ----------------------------------------#
fit_nnetar_sj <- nnetar(sj_time_series,repeats=25, size=12, decay=0.1,linout=TRUE)
plot(forecast(fit_nnetar_sj,h=260))
a = forecast(fit_nnetar_sj,h=260)


fit_nnetar_iq <- nnetar(iq_time_series,repeats=25, size=18, decay=0.1,linout=TRUE)
plot(forecast(fit_nnetar_iq,h=156))
b = forecast(fit_nnetar_iq,h=156)

sj.final3 <- data.frame(sj_testdata[,1:3],total_cases = round(a$mean))
iq.final3 <- data.frame(iq_testdata[,1:3],total_cases = round(b$mean))
final3 <- bind_rows(sj.final3,iq.final3)

write.csv(final2,"nnetar_model_final3.csv",row.names = F)

nrow(final3)
#--------------Random Forest--------------------------------------------#
sj_fit <- randomForest(total_cases ~
                               reanalysis_specific_humidity_g_per_kg +
                               reanalysis_dew_point_temp_k + 
                               station_avg_temp_c +
                               station_min_temp_c
                             , data = train_sj)

print(sj_fit)

sj_fit_predict <- predict(object=sj_fit, sj_testdata)

iq_fit_model <- randomForest(total_cases ~
                               reanalysis_specific_humidity_g_per_kg +
                               reanalysis_dew_point_temp_k + 
                               station_avg_temp_c +
                               station_min_temp_c
                             , data = train_iq)

print(iq_fit_model)

iq_fit_prediction<- predict(object=iq_fit_model, iq_testdata)
