using KoalaLightGBM
using Base.Test

using Koala
const X, y = load_boston();

const train, test = split(1:length(y), 0.9);
rgs = LGBMRegressor(validation_fraction = 0.2,
                    num_iterations=100,
                    num_leaves=2, min_data_in_leaf=12)

# change default transformer to treate two features as categorical
# (but see below):
tr_X = default_transformer_X(rgs)
tr_X.categorical_features = [:Rad, :Chas]

mach = Machine(rgs, X, y, train, transformer_X = tr_X)
fit!(mach, train)
rgs.validation_fraction = 0.0
fit!(mach, train)
score = err(mach, test)
println("error = $score")
@test score > 3 && score < 4

# instead change some to string type to declare categorical:
X[:Rad] = map(string,X[:Rad])
X[:Chas] = map(string,X[:Chas])

# no need to change default transformer:
mach = Machine(rgs, X, y, train)
fit!(mach, train)
rgs.validation_fraction = 0.0
fit!(mach, train)
score = err(mach, test)
println("error = $score")
@test score > 3 && score < 4
fit!(mach)


## CLASSIFICATION

# get some classification data:
const Xc, yc = load_iris() # has 3-class target


# restrict to binary target:
mask = yc .!= "versicolor"
Xc = Xc[mask,:]; yc = yc[mask]
const Y = map(yc) do t
    t == "setosa" ? 0 : 1
end

# randomize:
rows = StatsBase.sample(eachindex(Y), length(Y), replace=false)
Xc = Xc[rows,:]
Y = Y[rows]

# split rows into train and test and build model:
train, test = split(eachindex(Y), 0.6);
clf = LGBMBinaryClassifier(min_data_in_leaf=1, min_sum_hessian_in_leaf=0)
clfM = Machine(clf, Xc, Y, train)

clf.validation_fraction=0.2
fit!(clfM, train)

clf.validation_fraction=0.0
fit!(clfM)
predictions_on_test = map(predict(clfM, Xc, test)) do score
    score >= 0.5 ? 1 : 0
end
@test predictions_on_test == Y[test] # test for perfect accuracy

