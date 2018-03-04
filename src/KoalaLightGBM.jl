__precompile__()
module KoalaLightGBM

export LGBMRegressor

import Koala: Regressor
import Koala: params
import KoalaTransforms
import DataFrames: AbstractDataFrame
try
    import LightGBM
catch exception
    display(Base.md"Problem loading the module `LGBM` module. Perhaps
              Microsoft's LightGBM is not installed (install with
              `Pkg.clone(\"https://github.com/Allardvm/LightGBM.jl.git\")`
               or that the system environment `LIGHTGBM_PATH` has not been 
               set to its location. ")
    throw(exception)
end

# to be extended (but not explicitly rexported):
import Koala: setup, fit, predict
import Koala: default_transformer_X, default_transformer_y, transform, inverse_transform

# development only:
# import ADBUtilities: @dbg, @colon

"""
## `type LGBMRegressor`

See
!(https://github.com/Allardvm/LightGBM)[https://github.com/Allardvm/LightGBM]
for some details. For tuning see
![https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-Tuning.rst)

"""
mutable struct LGBMRegressor <: Regressor{LightGBM.LGBMRegression}

    num_iterations::Int                  # num_iterations in internal model
    learning_rate::Float64
    num_leaves::Int 
    max_depth::Int 
    tree_learner::String 
    num_threads::Int 
    histogram_pool_size::Float64
    min_data_in_leaf::Int              # aka min_patterns_split
    min_sum_hessian_in_leaf::Float64 
    feature_fraction::Float64 
    feature_fraction_seed::Int
    bagging_fraction::Float64
    bagging_freq::Int 
    bagging_seed::Int
    early_stopping_round::Int
    max_bin::Int 
    data_random_seed::Int 
    init_score::String 
    is_sparse::Bool 
    save_binary::Bool
    is_unbalance::Bool
    metric::Vector{String}
    metric_freq::Int 
    is_training_metric::Bool 
    ndcg_at::Vector{Int} 
    num_machines::Int 
    local_listen_port::Int 
    time_out::Int 
    machine_list_file::String 
    validation_fraction::Float64  # if zero then no validation errors
                                  # computed or reported

end

# lazy keyword constructor:
LGBMRegressor(;num_iterations=10, learning_rate=.1, num_leaves=127, max_depth=-1,
              tree_learner="serial", num_threads=Sys.CPU_CORES,
              histogram_pool_size=-1.,
              min_data_in_leaf=100, min_sum_hessian_in_leaf=10.,
              feature_fraction=1., feature_fraction_seed=0,
              bagging_fraction=1., bagging_freq=1, bagging_seed=0,
              early_stopping_round=4, max_bin=255,
              data_random_seed=0, init_score="", is_sparse=true,
              save_binary=false, is_unbalance=false, metric=["l2"],
              metric_freq=1, is_training_metric=false,
              ndcg_at=Int[], num_machines=1, local_listen_port=12400, time_out=120,
              machine_list_file="",
              validation_fraction=0.0) = LGBMRegressor(num_iterations, learning_rate,
                                                     num_leaves, max_depth,
                                                     tree_learner, num_threads,
                                                     histogram_pool_size,
                                                     min_data_in_leaf,
                                                     min_sum_hessian_in_leaf,
                                                     feature_fraction,
                                                     feature_fraction_seed,
                                                     bagging_fraction, bagging_freq,
                                                     bagging_seed,
                                                     early_stopping_round,
                                                     max_bin, data_random_seed,
                                                     init_score, is_sparse,
                                                     save_binary, is_unbalance,
                                                     metric, metric_freq,
                                                     is_training_metric, ndcg_at,
                                                     num_machines, local_listen_port,
                                                     time_out, machine_list_file,
                                                     validation_fraction)

default_transformer_X(model::LGBMRegressor) =
    KoalaTransforms.DataFrameToArrayTransformer()
default_transformer_y(model::LGBMRegressor) =
    KoalaTransforms.RegressionTargetTransformer()

function setup(rgs::LGBMRegressor,
               X::Matrix{T},
               y::Vector{T},
               features, parallel, verbosity) where T <: Real
    return X, y
end

function fit(rgs::LGBMRegressor, cache, add, parallel, verbosity)

    X, y, = cache

    # Microsoft's LightGBM has option for reporting running validation
    # scores; so we split the data if `validation_fraction` is bigger
    # than zero:
    train_fraction = 1 - rgs.validation_fraction
    if rgs.validation_fraction != 0.0
        train, valid = split(eachindex(y), train_fraction)
        Xvalid = X[valid,:]
        yvalid = y[valid]
        X = X[train,:]
        y = y[train]
    end

    parameters = params(rgs)
    delete!(parameters, :validation_fraction) # not sent to inner fit
    if rgs.feature_fraction_seed == 0
        parameters[:feature_fraction_seed] = round(Int, time())
    end
    if rgs.bagging_seed == 0
        parameters[:bagging_seed] = round(Int, time())
    end
    if rgs.data_random_seed == 0
        parameters[:data_random_seed] = round(Int, time())
    end
    if !parallel
        parameters[:num_threads] = 1
    end

    predictor = LightGBM.LGBMRegression(;parameters...)

    valid_pairs = Tuple{Matrix{Float64},Vector{Float64}}[]
    if rgs.validation_fraction != 0.0
        push!(valid_pairs, (Xvalid, yvalid))
    end
    output = values(LightGBM.fit(predictor, X, y, valid_pairs...;
                                 verbosity=verbosity)) |> collect

    report = Dict{Symbol,Array{Float64,1}}()

    if !isempty(output)
        report[:rms_raw_validation_errors] = output[1]["l2"]
    end
       
    return predictor, report, (X, y)
end

function predict(rgs::LGBMRegressor, predictor, X, parallel, verbosity)
    return LightGBM.predict(predictor, X, verbosity=0)
end

end # module

