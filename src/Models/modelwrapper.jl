############################################################################################
# Assign abstract types
"""
$(TYPEDEF)
Abstract super type for Baytes Models.
"""
abstract type ModelName end
"""
$(TYPEDEF)
Default modelname of Baytes.Model struct.
"""
struct BaseModel <: ModelName end

############################################################################################
#!NOTE: For M<:Union{P,ModelName} where {P}, keep A so option using of other PPLs is easily configurable.
"""
$(TYPEDEF)
Baytes Model struct.

Contains information about current Model value, name, and information, see also [`ParameterInfo`](@ref).

# Fields
$(TYPEDFIELDS)
"""
mutable struct ModelWrapper{
    M<:Union{P,ModelName} where {P},A<:NamedTuple, C<:NamedTuple, B<:ParameterInfo
} <: BaytesCore.AbstractModelWrapper
    "Current Model values as NamedTuple - works with Nested Tuples."
    val::A
    "Supplementary arguments for log target function that are fixed and dont need to be stored in a trace."
    arg::C
    "Information about parameter distributions, transformations and constraints, see [`ParameterInfo`](@ref)."
    info::B
    "Model id, per default BaseModel. Useful for dispatching ModelWrapper struct."
    id::M
    function ModelWrapper(
        val::A, arg::C, info::B, id::M
    ) where {M<:Union{P,ModelName} where {P},A<:NamedTuple,C<:NamedTuple,B<:ParameterInfo}
        return new{M,A,C,B}(val, arg, info, id)
    end
end
# Convenient Constructor
function ModelWrapper(
    id::M, parameter::A, arg::C=(;), flattendefault::F=FlattenDefault()
) where {M<:Union{P,ModelName} where {P},A<:NamedTuple,C<:NamedTuple,F<:FlattenDefault}
    ## Create ParameterInfo struct
    paraminfo = ParameterInfo(flattendefault, parameter)
    ## Split between values and constraints
    val = _get_val(parameter)
    ## Return ModelWrapper
    return ModelWrapper(val, arg, paraminfo, id)
end
ModelWrapper(parameter::A, arg::C=(;), flattendefault::F=FlattenDefault()) where {A<:NamedTuple,C<:NamedTuple,F<:FlattenDefault} =
    ModelWrapper(BaseModel(), parameter, arg, flattendefault)

############################################################################################
# Basic functions for Model struct
length_constrained(model::ModelWrapper) = model.info.reconstruct.unflatten.strict._unflatten.sz[end]
length_unconstrained(model::ModelWrapper) = model.info.reconstructᵤ.unflatten.strict._unflatten.sz[end]

paramnames(model::ModelWrapper) = keys(model.val)

############################################################################################
# A bunch of functions that can be used/extended for target model in Sampling process
"""
$(SIGNATURES)
Simulate data given Model parameter.

# Examples
```julia
```

"""
function simulate end

############################################################################################
# Dispatch Model struct for .Core functions

"""
$(SIGNATURES)
Show current values of Model as NamedTuple

# Examples
```julia
```

"""
function generate_showvalues(model::ModelWrapper)
    return function showvalues()
        return ((:Parameter, model.val), )
    end
end

"""
$(SIGNATURES)
Fill 'model' values with NamedTuple 'θ'.

# Examples
```julia
```

"""
function fill(model::ModelWrapper, θ::NamedTuple)
    return merge(model.val, θ)
end

"""
$(SIGNATURES)
Inplace version of [`fill`](@ref).

# Examples
```julia
```

"""
function fill!(model::ModelWrapper, θ::NamedTuple)
    model.val = merge(model.val, θ)
    return nothing
end

function subset(model::ModelWrapper, sym)
    return subset(model.val, sym)
end

#########################################
"""
$(SIGNATURES)
Unconstrain 'model' values and return as NamedTuple.

# Examples
```julia
```

"""
function unconstrain(model::ModelWrapper)
    return unconstrain(model.info, model.val)
end

"""
$(SIGNATURES)
Constrain 'θᵤ' values with model.info ParameterInfo.

# Examples
```julia
```

"""
function constrain(model::ModelWrapper, θ::NamedTuple)
    return constrain(model.info, θ)
end

"""
$(SIGNATURES)
Flatten 'model' values and return as vector.

# Examples
```julia
```

"""
function flatten(model::ModelWrapper)
    return flatten(model.info, model.val)
end

"""
$(SIGNATURES)
Flatten and unconstrain 'model' values and return as vector.

# Examples
```julia
```

"""
function unconstrain_flatten(model::ModelWrapper)
    return unconstrain_flatten(model.info, model.val)
end
function unconstrain_flattenAD(model::ModelWrapper)
    return unconstrain_flattenAD(model.info, model.val)
end

#########################################
"""
$(SIGNATURES)
Unlatten Vector 'θ' given constraints from 'model' and return as NamedTuple.

# Examples
```julia
```

"""
function unflatten(model::ModelWrapper, θ::AbstractVector{T}) where {T<:Real}
    return unflatten(model.info, θ)
end

"""
$(SIGNATURES)
Inplace version of [`unflatten`](@ref).

# Examples
```julia
```

"""
function unflatten!(model::ModelWrapper, θ::AbstractVector{T}) where {T<:Real}
    model.val = unflatten(model, θ)
    return nothing
end

"""
$(SIGNATURES)
Constrain and Unflatten vector 'θᵤ' given 'model' constraints.

# Examples
```julia
```

"""
function unflatten_constrain(model::ModelWrapper, θᵤ::AbstractVector{T}) where {T<:Real}
    return unflatten_constrain(model.info, θᵤ)
end
function unflattenAD_constrain(model::ModelWrapper, θᵤ::AbstractVector{T}) where {T<:Real}
    return unflattenAD_constrain(model.info, θᵤ)
end

"""
$(SIGNATURES)
Inplace version of [`unflatten_constrain`](@ref).

# Examples
```julia
```

"""
function unflatten_constrain!(model::ModelWrapper, θᵤ::AbstractVector{T}) where {T<:Real}
    model.val = unflatten_constrain(model, θᵤ)
    return nothing
end

#########################################
"""
$(SIGNATURES)
Sample from 'model' prior and return as NamedTuple.

# Examples
```julia
```

"""
function sample(_rng::Random.AbstractRNG, model::ModelWrapper)
    return sample_constraint(_rng, model.info.transform.constraint, model.val)
end
sample(model::ModelWrapper) = sample(Random.GLOBAL_RNG, model)

"""
$(SIGNATURES)
Inplace version of [`sample`](@ref).

# Examples
```julia
```

"""
function sample!(_rng::Random.AbstractRNG, model::ModelWrapper)
    model.val = sample(_rng, model)
    return nothing
end
sample!(model::ModelWrapper) = sample!(Random.GLOBAL_RNG, model)

#########################################
"""
$(SIGNATURES)
Evaluate Log density of 'model' prior given current 'model' values.

# Examples
```julia
```

"""
function log_prior(model::ModelWrapper)
    return log_prior(model.info.transform.constraint, model.val)
end

"""
$(SIGNATURES)
Evaluate Log density and eventual Jacobian adjustments of 'model' prior given current 'model' values.

# Examples
```julia
```

"""
function log_prior_with_transform(model::ModelWrapper)
    return log_prior_with_transform(model.info.transform.constraint, model.val)
end

"""
$(SIGNATURES)
Evaluate eventual Jacobian adjustments from transformations at 'model' values.

# Examples
```julia
```

"""
function log_abs_det_jac(model::ModelWrapper)
    return log_abs_det_jac(model.info, model.val)
end

#########################################
function print(sym::Symbol, val, constraint)
    println("#############################################################")
    println("Parameter ", sym)
    println("Dimensionality: ", size(val))
    println("Constraint: ", typeof(constraint))
    println("Value: ", val)
end

"""
$(SIGNATURES)
Print 'model' parameter values and constraints of symbols 'params'.

# Examples
```julia
```

"""
function print(model::ModelWrapper, params::S = keys(model.val)) where {S<:Union{Symbol,NTuple{k,Symbol} where k}}
    for sym in params
        val = getfield(model.val, sym)
        constraint = getfield(model.info.transform.constraint, sym)
        print(sym, val, constraint)
    end
end

############################################################################################
export
    ModelName,
    BaseModel,
    ModelWrapper,
    simulate,
    length_constrained,
    length_unconstrained,
    paramnames,
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
    log_abs_det_jac,
    print
