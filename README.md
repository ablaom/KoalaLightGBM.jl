# KoalaLightGBM

Microsoft's gradient boosting machine learning package
[LightGBM](https://github.com/Microsoft/LightGBM) is here wrapped for
use in the [Koala](https://github.com/ablaom/Koala.jl) machine
learning environment. It is built entirely on top of Allard van Mossel's
julia wrap [LightGBM.jl](https://github.com/Allardvm/LightGBM.jl)
of the Microsoft software.

## Usage example:

In the following example we start with the hyperparameter `validation_fraction=0.2` to obtain a running validation score for the training run. This means 20% of the data in rows `train` is held back in training for the validation scoring. Then we train on the full `train` set

````julia
    julia> using KoalaLightGBM
	julia> using Koala
	julia> const X, y = load_boston();
	julia> const train, test = splitrows(1:length(y), 0.8); # 80:20 split
	julia> rgs = LGBMRegressor(validation_fraction = 0.2,
    num_iterations=100,
    num_leaves=2, min_data_in_leaf=12)
	LGBMRegressor@...312

	julia> mach = SupervisedMachine(rgs, X, y, train)
	SupervisedMachine{LGBMRegressor,}@...392

	julia> fit!(mach, train)
	SupervisedMachine{LGBMRegressor,}@...392

	julia> mach.report
    Dict{Symbol,Array{Float64,1}} with 1 entry:
     :rms_raw_validation_errors => [1.53358, 1.4901, 1.46696, 1.42826, 1.41359,…

	julia> score = err(mach, test)
	5.626761247184338
	
	julia> rgs.validation_fraction=0 # so fit! trains on full train set
	julia> fit!(mach, train)
    SupervisedMachine{LGBMRegressor,}@...392

    julia> err(mach, test)
    3.6908332199941105
	````


