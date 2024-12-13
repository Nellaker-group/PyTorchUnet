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
val_total<-res[ grepl("val:",res$V1) & grepl("TOTAL",res$V2),]


bitmap(paste0(prefix,"_lossPlotV3.png"),res=300)
layout(matrix(c(1,1,1,2,1,1,1,2,3,3,3,4,3,3,3,4), 4, 4, byrow = F))

minLoss<-round(min(val_total[,"V7"]),2)
w<-which.min(val_total[,"V7"])
minLossDice<-round(1-val_total[w,"V5"],2)

ymax<-max(c(res[ grepl("train",res$V1),"V7"],res[  grepl("val:",res$V1) & grepl("TOTAL",res$V2),"V7"]))
ymin<-min(c(res[ grepl("train",res$V1),"V7"],res[  grepl("val:",res$V1) & grepl("TOTAL",res$V2),"V7"]))


plot(1:epochs, res[ grepl("train",res$V1),"V7"],type="l",xlab="epochs",ylab="loss",ylim=c(0,2),col="red",main=paste0("loss, min loss=",minLoss,", min Loss Dice=",minLossDice)); 
lines(1:epochs, res[ grepl("TOTAL",res$V2) ,"V7"],col="blue")
lines(1:epochs, res[ grepl("endox",res$V2),"V7"],col="grey")
lines(1:epochs, res[ grepl("gtex",res$V2),"V7"],col="green")
lines(1:epochs, res[ grepl("hohen",res$V2),"V7"],col="black")
lines(1:epochs, res[ grepl("leipzig",res$V2),"V7"],col="purple")
lines(1:epochs, res[ grepl("munich",res$V2),"V7"],col="gold")

legend("topright",c("train TOTAL","val TOTAL","val endox","val gtex", "val hohen", "val leipzig", "val munich"),fill=c("red","blue","grey","green","black","purple","gold"))

plot(1:epochs, log10(res[ grepl("train",res$V1),"V9"]),type="l",xlab="epochs",ylab="log10(LR)",col="black",main="learning rate (LR)")
par(xpd=T)
for(u in unique(val_total$V9)){
      text(paste0("LR ",u),x=median(which(val_total$V9%in%u)),y=log10(u)-log10(u)/20)
}
par(xpd=F)


maxDice<-round(max(1-val_total[,"V5"]),2)

ymax<-max(c(1-res[ grepl("train",res$V1),"V5"],res[ grepl("val:",res$V1) & grepl("TOTAL",res$V2),"V5"]))
ymin<-min(c(1-res[ grepl("train",res$V1),"V5"],res[ grepl("val:",res$V1) & grepl("TOTAL",res$V2),"V5"]))
plot(1:epochs, 1-res[ grepl("train",res$V1),"V5"],type="l",xlab="epochs",ylab="dice coef",ylim=c(0.2,1),col="red",main=paste0("dice coef, max Dice=",maxDice))
lines(1:epochs, 1-res[ grepl("TOTAL",res$V2),"V7"],col="blue")
lines(1:epochs, 1-res[ grepl("endox",res$V2),"V7"],col="grey")
lines(1:epochs, 1-res[ grepl("gtex",res$V2),"V7"],col="green")
lines(1:epochs, 1-res[ grepl("hohen",res$V2),"V7"],col="black")
lines(1:epochs, 1-res[ grepl("leipzig",res$V2),"V7"],col="purple")
lines(1:epochs, 1-res[ grepl("munich",res$V2),"V7"],col="gold")
legend("topright",c("train TOTAL","val TOTAL","val endox","val gtex", "val hohen", "val leipzig", "val munich"),fill=c("red","blue","grey","green","black","purple","gold"))

plot(1:epochs, log10(res[ grepl("train",res$V1),"V9"]),type="l",xlab="epochs",ylab="log10(LR)",col="black",main="learning rate (LR)")
par(xpd=T)
for(u in unique(val_total$V9)){
      text(paste0("LR ",u),x=median(which(val_total$V9%in%u)),y=log10(u)-log10(u)/20)
}
par(xpd=F)

dev.off()


print("min loss")
print(min(val_total[,"V7"]))
w<-which.min(val_total[,"V7"])
print("min loss has this dice coef")
print(1-val_total[w,"V5"])

print("max dice coef")
print(max(1-val_total[,"V5"]))
w<-which.max(1-val_total[,"V5"])
print(w)
print("max dice coef has this loss")
print(val_total[w,"V7"])

