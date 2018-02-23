using KoalaLightGBM
using Base.Test

using Koala
const X, y = load_boston();
const train, test = splitrows(1:length(y), 0.8);
rgs = LGBMRegressor(validation_fraction = 0.2,
                    num_iterations=100,
                    num_leaves=2, min_data_in_leaf=12)
mach = Machine(rgs, X, y, train)
fit!(mach, train)
rgs.validation_fraction = 0.0
fit!(mach, train)
score = err(mach, test)
println("error = $score")
@test score > 3 && score < 4
