function accuracy = lda_mle_classifier_ding(data,fea_sel)

d_length = length(data);

%% divide data to train and test set
train_data = data(1:d_length/2,:);
test_data = data(d_length/2+1:end,:);

label0 = train_data(:,9)==0;
label1 = train_data(:,9)==1;

%% operate the active feature user selected

train_fea = train_data(:,fea_sel);
test_fea  = test_data(:,fea_sel);

train_fea_0 = train_fea(label0 == 1,:);
train_fea_1 = train_fea(label1 == 1,:);

[mean_0_raw,sigma_0_raw] = mle_ding(train_fea_0);
[mean_1_raw,sigma_1_raw] = mle_ding(train_fea_1);

n0 = length(train_fea_0);
n1 = length(train_fea_1);

S0 = n0*sigma_0_raw;
S1 = n1*sigma_1_raw;

Sw = S0+S1;

Sw_inv = inv(Sw);

v = Sw_inv*(mean_0_raw-mean_1_raw)';

%% compute the projected vector
train_fea_0_lda = v'*train_fea_0';
train_fea_1_lda = v'*train_fea_1';
% compute the mean and variance of projected training sample 
[mean_0,sigma_0] = mle_ding(train_fea_0_lda);
[mean_1,sigma_1] = mle_ding(train_fea_1_lda);

test_fea_lda = (v'*test_fea')';

%% compute the prior of those feature
prior0_num = length(train_fea_0);
prior1_num = length(train_fea_1);

prior0 = prior0_num/(prior0_num+prior1_num);
prior1 = 1- prior0;

%% test the test feature set
correct = 0;
wrong = 0;

for i=1:length(test_data)
    likelihood0 = exp(-(test_fea_lda(i,:)-mean_0)/(2*sigma_0)*(test_fea_lda(i,:)-mean_0)')/sqrt(det(sigma_0));
    likelihood1 = exp(-(test_fea_lda(i,:)-mean_1)/(2*sigma_1)*(test_fea_lda(i,:)-mean_1)')/sqrt(det(sigma_1));
    post0 = likelihood0*prior0/(likelihood0*prior0+likelihood1*prior1);
    if(post0 > 0.5 && test_data(i,9) == 0)
       correct = correct+1;
    elseif ( post0 <= 0.5 &&test_data(i,9) == 1)
        correct = correct+1;
    else
        wrong = wrong+1;
    end
end
accuracy = 1 - wrong/length(test_data);


