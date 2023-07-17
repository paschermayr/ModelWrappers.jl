############################################################################################
"""
$(TYPEDEF)
Contains information about parameter distributions, transformations and constraints.

# Fields
$(TYPEDFIELDS)
"""
struct ParameterInfo{R<:ReConstructor, U<:ReConstructor, T<:TransformConstructor}
    "Contains information for flatten/unflatten parameter"
    reconstruct::R
    "Contains information to reconstruct unconstrained parameter - important for non-bijective transformations"
    reconstructᵤ::U
    "Contains information for constraining and unconstraining parameter."
    transform::T
    function ParameterInfo(
        reconstruct::R, reconstructᵤ::U, transform::T
    ) where {R<:ReConstructor, U<:ReConstructor, T<:TransformConstructor}
        return new{R, U, T}(
            reconstruct, reconstructᵤ, transform
        )
    end
end
function ParameterInfo(flattendefault::D, constructor::R, transformer::T, val::V) where {D<:FlattenDefault, R<:ReConstructor, T<:TransformConstructor, V}
    ## Construct flatten constructor for unconstrained parameterization - important for non-bijective transformations 
    constructorᵤ = ReConstructor(flattendefault, transformer.constraint, unconstrain(transformer, val))
    return ParameterInfo(
        constructor, constructorᵤ, transformer
    )
end

function ParameterInfo(
    flattendefault::D, constraint::C, val::B
) where {D<:FlattenDefault, C<:NamedTuple, B<:NamedTuple}
    ## Create flatten constructor
    constructor = ReConstructor(flattendefault, constraint, val)
    ## Assign transformer constraint NamedTuple
    transformer = TransformConstructor(constraint, val)
    ## Construct flatten constructor for unconstrained parameterization - important for non-bijective transformations 
    constructorᵤ = ReConstructor(flattendefault, constraint, unconstrain(transformer, val))
    ## Return ParameterInfo
    return ParameterInfo(
        constructor, constructorᵤ, transformer
    )
end
function ParameterInfo(
    flattendefault::D, parameter::B
) where {D<:FlattenDefault, B<:NamedTuple}
    ## Check if all values in val are of type Param
    ArgCheck.@argcheck _checkparams(parameter) "All values in (nested) NamedTuple have to be of Type Param."
    ## Split between values and constraints
    val = _get_val(parameter)
    constraint = _get_constraint(parameter)
    return ParameterInfo(
        flattendefault, constraint, val
    )
end

############################################################################################
length_constrained(info::ParameterInfo) = info.reconstruct.unflatten.strict._unflatten.sz[end]
length_unconstrained(info::ParameterInfo) = info.reconstructᵤ.unflatten.strict._unflatten.sz[end]


############################################################################################
function flatten(info::ParameterInfo, x)
    return flatten(info.reconstruct, x)
end
function flattenAD(info::ParameterInfo, x)
    return flattenAD(info.reconstruct, x)
end
function unflatten(info::ParameterInfo, x)
    return unflatten(info.reconstruct, x)
end
function unflattenAD(info::ParameterInfo, x)
    return unflattenAD(info.reconstruct, x)
end

############################################################################################
function constrain(info::ParameterInfo, valᵤ::V) where {V}
    return constrain(info.transform, valᵤ)
end
function unconstrain(info::ParameterInfo, val::V) where {V}
    return unconstrain(info.transform, val)
end
function log_abs_det_jac(info::ParameterInfo, val::V) where {V}
    return log_abs_det_jac(info.transform, val)
end

function unconstrain_flatten(info::ParameterInfo, val::V) where {V}
    return flatten(info.reconstructᵤ, unconstrain(info.transform, val))
end
function unconstrain_flattenAD(info::ParameterInfo, val::V) where {V}
    return flattenAD(info.reconstructᵤ, unconstrain(info.transform, val))
end

function unflatten_constrain(info::ParameterInfo, valᵤ::V) where {V}
    return constrain(info.transform, unflatten(info.reconstructᵤ, valᵤ))
end
function unflattenAD_constrain(info::ParameterInfo, valᵤ::V) where {V}
    return constrain(info.transform, unflattenAD(info.reconstructᵤ, valᵤ))
end

############################################################################################
#export
export ParameterInfo

