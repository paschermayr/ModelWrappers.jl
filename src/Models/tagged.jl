############################################################################################
"""
$(TYPEDEF)

Stores information for a subset of 'model' parameter.

# Fields
$(TYPEDFIELDS)
"""
struct Tagged{A<:NamedTuple,B<:ParameterInfo}
    "Subset of ModelWrapper parameter names."
    parameter::A
    "Information about subset of parameter distributions, transformations and constraints, see [`ParameterInfo`](@ref)."
    info::B
    function Tagged(
        model::ModelWrapper, sym::S=keys(model.val)
    ) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
        ## If sym is a single symbol, convert to tuple
        if isa(sym, Symbol)
            sym = (sym,)
        end
        ## Check if all symbols are contained in ModelWrapper
        ArgCheck.@argcheck all(haskey(model.val, param) for param in sym) "Cannot tag parameter that is not contained in ModelWrapper.val"
        ## Convert Tuple to NamedTuple with fields == value of tuple -> can subset without allocations
        sym = Tuple_to_Namedtuple(sym, true)
        ## Generate new ParameterInfo based on sym subset
        info = ParameterInfo(
            subset(model.info.constraint, sym),
            subset(model.val, sym),
            model.info.reconstruct.default,
        )
        ## Return Tagged
        return new{typeof(sym),typeof(info)}(sym, info)
    end
end

############################################################################################
# Basic functions for Tagged struct
length(tagged::Tagged) = tagged.info.reconstruct.unflatten.strict._unflatten.sz[end]
paramnames(tagged::Tagged) = keys(tagged.parameter)

#A convenient method for evaluating a prior distribution of a NamedTuple parameter
function log_prior(tagged::Tagged, θ::NamedTuple)
    return log_prior(tagged.info.constraint, subset(θ, tagged.parameter))
end

############################################################################################
# Dispatch Tagged struct for .Core functions
function generate_showvalues(model::ModelWrapper, tagged::Tagged)
    return function showvalues()
        return ((:Parameter, subset(model, tagged)), )
    end
end
function fill(model::ModelWrapper, tagged::Tagged, θ::NamedTuple)
    return merge(model.val, subset(θ, tagged.parameter))
end
function fill!(model::ModelWrapper, tagged::Tagged, θ::NamedTuple)
    model.val = merge(model.val, subset(θ, tagged.parameter))
    return nothing
end
function subset(model::ModelWrapper, tagged::Tagged)
    return subset(model.val, tagged.parameter)
end

#########################################
function unconstrain(model::ModelWrapper, tagged::Tagged)
    return unconstrain(tagged.info.transform, subset(model, tagged))
end
function flatten(model::ModelWrapper, tagged::Tagged)
    return flatten(tagged.info.reconstruct, subset(model, tagged))
end
function unconstrain_flatten(model::ModelWrapper, tagged::Tagged)
    return flatten(tagged.info.reconstruct, unconstrain(model, tagged))
end

#########################################
function unflatten(
    model::ModelWrapper, tagged::Tagged, θ::AbstractVector{T}
) where {T<:Real}
    return unflatten(tagged.info.reconstruct, θ)
end
function unflatten!(
    model::ModelWrapper, tagged::Tagged, θ::AbstractVector{T}
) where {T<:Real}
    model.val = merge(model.val, unflatten(model, tagged, θ))
    return nothing
end
function unflatten_constrain(
    model::ModelWrapper, tagged::Tagged, θᵤ::AbstractVector{T}
) where {T<:Real}
    return constrain(tagged.info.transform, unflatten(model, tagged, θᵤ))
end
function unflatten_constrain!(
    model::ModelWrapper, tagged::Tagged, θᵤ::AbstractVector{T}
) where {T<:Real}
    model.val = merge(model.val, unflatten_constrain(model, tagged, θᵤ))
    return nothing
end

#########################################
function sample(_rng::Random.AbstractRNG, model::ModelWrapper, tagged::Tagged)
    return merge(model.val, sample_constraint(_rng, tagged.info.constraint, subset(model, tagged)))
end
sample(model::ModelWrapper, tagged::Tagged) = sample(Random.GLOBAL_RNG, model, tagged)

function sample!(_rng::Random.AbstractRNG, model::ModelWrapper, tagged::Tagged)
    ArgCheck.@argcheck _checkprior(subset(tagged.info.constraint, tagged.parameter)) "For inplace sample version, all constraints need to be a Distribution."
    model.val = sample(_rng, model, tagged)
    return nothing
end
sample!(model::ModelWrapper, tagged::Tagged) = sample!(Random.GLOBAL_RNG, model, tagged)

#########################################
function log_prior(model::ModelWrapper, tagged::Tagged)
    return log_prior(tagged.info.constraint, subset(model, tagged))
end
function log_prior_with_transform(model::ModelWrapper, tagged::Tagged)
    return log_prior_with_transform(tagged.info.constraint, subset(model, tagged))
end
function log_abs_det_jac(model::ModelWrapper, tagged::Tagged)
    return log_abs_det_jac(tagged.info.transform.unconstrain, subset(model, tagged))
end

############################################################################################
export Tagged,
    length,
    fill,
    fill!,
    subset,
    flatten,
    unconstrain_flatten,
    unflatten,
    unflatten!,
    unconstrain,
    unflatten_constrain,
    unflatten_constrain!,
    sample,
    sample!,
    log_prior,
    log_prior_with_transform,
    log_abs_det_jac
