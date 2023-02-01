name <- commandArgs(trailingOnly = TRUE)[1]

res<-read.table(name,as.is=T,h=F)

prefix<-sub(".res","",name)
prefix<-sub(".log","",prefix)

comma<-function(x) { return(as.numeric(sub(",","",x))) }

epochs <- length(res[ grepl("train",res$V1),"V7"])

res$V3<-comma(res$V3)
res$V5<-comma(res$V5)
res$V7<-comma(res$V7)

val<-res[ grepl("val",res$V1),]



bitmap(paste0(prefix,"_lossPlotV2.png"),res=300)
layout(matrix(c(1,1,1,2,1,1,1,2,3,3,3,4,3,3,3,4), 4, 4, byrow = F))



minLoss<-round(min(val[,"V7"]),2)
w<-which.min(val[,"V7"])
minLossDice<-round(1-val[w,"V5"],2)


ymax<-max(c(res[ grepl("train",res$V1),"V7"],res[ grepl("val",res$V1),"V7"]))
ymin<-min(c(res[ grepl("train",res$V1),"V7"],res[ grepl("val",res$V1),"V7"]))
plot(1:epochs, res[ grepl("train",res$V1),"V7"],type="l",xlab="epochs",ylab="loss",ylim=c(0,2),col="red",main=paste0("loss, min loss=",minLoss,", min Loss Dice=",minLossDice)); 
lines(1:epochs, res[ grepl("val",res$V1),"V7"],col="blue")
legend("topright",c("val","train"),fill=c("blue","red"))

plot(1:epochs, log10(res[ grepl("train",res$V1),"V9"]),type="l",xlab="epochs",ylab="log10(LR)",col="black",main="learning rate (LR)")
par(xpd=T)
for(u in unique(val$V9)){
      text(paste0("LR ",u),x=median(which(val$V9%in%u)),y=log10(u)-log10(u)/20)
}
par(xpd=F)

maxDice<-round(max(1-val[,"V5"]),2)

ymax<-max(c(1-res[ grepl("train",res$V1),"V5"],res[ grepl("val",res$V1),"V5"]))
ymin<-min(c(1-res[ grepl("train",res$V1),"V5"],res[ grepl("val",res$V1),"V5"]))
plot(1:epochs, 1-res[ grepl("train",res$V1),"V5"],type="l",xlab="epochs",ylab="dice coef",ylim=c(0.4,1),col="red",main=paste0("dice coef, max Dice=",maxDice))
lines(1:epochs, 1-res[ grepl("val",res$V1),"V5"],col="blue")
legend("topright",c("val","train"),fill=c("blue","red"))

plot(1:epochs, log10(res[ grepl("train",res$V1),"V9"]),type="l",xlab="epochs",ylab="log10(LR)",col="black",main="learning rate (LR)")
par(xpd=T)
for(u in unique(val$V9)){
      text(paste0("LR ",u),x=median(which(val$V9%in%u)),y=log10(u)-log10(u)/20)
}
par(xpd=F)

dev.off()


print("min loss")
print(min(val[,"V7"]))
w<-which.min(val[,"V7"])
print("min loss has this dice coef")
print(1-val[w,"V5"])

print("max dice coef")
print(max(1-val[,"V5"]))
w<-which.max(1-val[,"V5"])
print(w)
print("max dice coef has this loss")
print(val[w,"V7"])

