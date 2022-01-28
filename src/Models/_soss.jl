############################################################################################
using Soss: Soss, ConditionalModel
import Soss: Soss, predict, simulate

############################################################################################
"Wrapper functions to work with Soss Models"
#=
Note: Some improvements to work on:
    -> In Objective (if multiple sampler are used to target only subset of parameter):
		When Objective is formed, make posterior such that only tagged parameter are evaluated
			-> would need to change old model.id in that case
    -> Make Methods for:
        simulate
        generate
        Predict -> predict(m(), (namedtuplevals))
	-> For Sequential Estimation methods (SMC and beyond)
        - Need to update data in SOSS model separately (when created as input?)
        - Need a way to separate log-prior and log-likelihood
=#

############################################################################################
# ModelWrapper part
"""
$(SIGNATURES)
Best guess for Soss Model parameter, excluding data and hyperparameter. Not exported.

# Examples
```julia
```

"""
function _guess_soss_param(soss_posterior::M) where {M<:Soss.ConditionalModel}
    ## All parameter from posterior.model.dists, except data (Hyperparameter should be fixed and have no model.dists entry)
    param = setdiff(keys(soss_posterior.model.dists), keys(soss_posterior.obs))
    ## Check if all param Symbols have values assigned
    ArgCheck.@argcheck all(haskey(soss_posterior.argvals, sym) for sym in param) "Not all posterior Soss model parameter have initial value assigned, please create initial value for all of them"
    ## Return NamedTuple with initial parameter
    return subset(soss_posterior.argvals, tuple(param...))
end

############################################################################################
function ModelWrapper(
    soss_posterior::M, flattendefault::F=FlattenDefault()
) where {M<:Soss.ConditionalModel,F<:FlattenDefault}
    ## Check if Posterior Soss Model provided
    ArgCheck.@argcheck !isempty(soss_posterior.obs) "No posterior Soss model provided, please use: posterior = MyModel | (MyDataName = MyDataValues,)"
    ## Guess all parameter
    val = _guess_soss_param(soss_posterior)
    ## Create prior struct from val
    _prior = subset(soss_posterior.model.dists, val)
    _prior_eval = NamedTuple{keys(_prior)}(eval(_prior[sym]) for sym in keys(_prior))
    ## Create Param NamedTuple as Safetye check that all defined Soss parameter are consistent with Param syntax
    #!NOTE: This is not ideal because all information was already there, but temporarily converting (val, prior) to Param struct guarantees that ModelWrapper can handle user input.
    params = NamedTuple{keys(val)}(
        Param(val[iter], _prior_eval[iter]) for iter in eachindex(val)
    )
    ## Return ModelWrapper
    return ModelWrapper(soss_posterior, params, flattendefault)
end

#=
#!NOTE: Could use predict(posterior, vals) directly as it returns data dimension. However, simualate needed for any algorithms and definitons slightly different in packages, so we leave it blank for now.
function simulate(model::ModelWrapper{M}) where {M<:Soss.ConditionalModel}
end
=#

############################################################################################
# Objective part

#!TODO: Make posterior such that only tagged parameter are evaluated
function Objective(model::ModelWrapper{M}) where {M<:Soss.ConditionalModel}
    return Objective(model::ModelWrapper{M}, nothing, Tagged(model))
end
function Objective(
    model::ModelWrapper{M}, tagged::T
) where {M<:Soss.ConditionalModel,T<:Tagged}
    return Objective(model::ModelWrapper{M}, nothing, tagged)
end

function (objective::Objective{<:ModelWrapper{M}})(
    θ::NamedTuple
) where {M<:Soss.ConditionalModel}
    return Soss.logdensity(objective.model.id(θ))
end

#!TODO: Need a way to update logposterior (model.id) with new data
#=

=#

#=
#NOTE: Waiting for: https://github.com/cscherrer/Soss.jl/issues/301
function predict(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Soss.ConditionalModel}
	return nothing
end
function generate(_rng::Random.AbstractRNG, objective::Objective{<:ModelWrapper{M}}) where {M<:Soss.ConditionalModel}
	return nothing
end
=#

############################################################################################
# Export
export ModelWrapper
