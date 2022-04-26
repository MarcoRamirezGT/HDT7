#Support Vector Machine

library(e1071)
library(caret)


data<-read.csv('train.csv')



data[is.na(data)] <- 0
#Calculo de percentiles
percentil <- quantile(data$SalePrice)
#Percentiles
estado<-c('Estado')
data$Estado<-estado
#Economica=0
#Intermedia=1
#Cara=2
data <- within(data, Estado[SalePrice<=129975] <- 'Economica')
data$Estado[(data$SalePrice>129975 & data$SalePrice<=163000)] <- 'Intermedia'
data$Estado[data$SalePrice>163000] <- 'Cara'

#Cambio de tipo de columnas
data$SalePrice<-as.numeric(data$SalePrice)
data$GrLivArea<-as.numeric(data$GrLivArea)
data$GarageCars<-as.numeric(data$GarageCars)
data$LotArea<-as.numeric(data$LotArea)
data$Estado<-as.factor(data$Estado)


#SVM
porcentaje<-0.7
set.seed(123)
datos<-data.frame(data$SalePrice,data$GrLivArea,data$GarageCars,data$LotArea,data$Estado)

#Datos de entrenamiento y prueba
corte <- sample(nrow(datos),nrow(datos)*porcentaje)
train<-datos[corte,]
test<-datos[-corte,]


modelosvm<-svm(data.Estado~., data = train, scale = F)


summary(modelosvm)
modelosvm$index
plot(modelosvm,train,data.SalePrice~data.GrLivArea)



modeloSVM_L<-svm(data.Estado~., data=train, cost=0.5, kernel="linear")#95%
modeloSVM_R<-svm(data.Estado~., data=train, gamma=2^-5, kernel="radial")
modeloSVM_R<-svm(data.Estado~., data=train, gamma=2^1, kernel="radial")


prediccionL<-predict(modeloSVM_L,newdata=test[,1:4])
prediccionR<-predict(modeloSVM_R,newdata=test[,1:4])


modeloTuneado<-tune.svm(data.Estado~., data=train, cost=c(0.01,0.1,0.5,1,5,10,16,20,32), kernel="linear")
predMejorModelo<-predict(modeloTuneado$best.model,newdata = test[,1:4])

confusionMatrix(test$data.Estado,prediccionL)
confusionMatrix(test$data.Estado,prediccionR)
confusionMatrix(test$data.Estado,predMejorModelo)


#Regresion lineal
fitLMPW<-lm(data.SalePrice~ ., data = train)
predL<-predict(fitLMPW, newdata = test)
#Verificando la predicci?n
resultados<-data.frame(test$data.SalePrice,predL)
head(resultados, n=5)
ggplot(data=train,mapping = aes(x=data.SalePrice,y=data.GrLivArea ))+
  geom_point(color='red',size=2)+
  geom_smooth(method = 'lm',se=TRUE,color='black')+
  labs(title = 'Precio de venta ~ Pies cuadrados de vivienda',x="Precio de venta",y='Pies cuadrados de vivienda')+
  theme_bw()+theme(plot.title = element_text(hjust = 0.5))


# #Overfitting
# library(ModelMetrics)
# 
# ##Modelo con todas las variables.
# modelo<-glm(data.Estado~., data = train,family = binomial(), maxit=100)
# pred<-predict(modelo,newdata = test, type = "response")
# prediccion<-ifelse(pred>=0.5,1,0)
# confusionMatrix(as.factor(test$data.Estado),as.factor(prediccion))
# 
# #Modelo para verificar overfitting
# 
# trainPredict<-predict(modelo,newdata = train, type = "response")
# trainPred<- ifelse(trainPredict>0.5,1,0)
# confusionMatrix(as.factor(train$data.Estado),as.factor(trainPred))
# 
# #Calculo de rmse para ver si tenemos overfittin, mientras mas cercano a 0 mayor overffiting.
# rmse(train$data.Estado,trainPred)
# rmse(test$data.Estado,prediccion)
# 
