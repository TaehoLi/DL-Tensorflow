clear all;
load trainlabels1.txt -ascii
load testlabels1.txt -ascii
%trainlabels(1:10)
Y=trainlabels1-1;
Yt=testlabels1-1;
save('trainlabels0.txt', 'Yt','-ascii')
save('testlabels0.txt', 'Yt','-ascii')
cccccc
Y=zeros(5000,10);
for i=1:5000,
Y(i, trainlabels1(i))=1;
end;

Yt=zeros(1000,10);
for i=1:1000,
Yt(i, testlabels1(i))=1;
end;

save( 'trainlabels.txt', 'Y','-ascii')
save('testlabels.txt', 'Yt','-ascii')