############################################################################################
"""
$(TYPEDEF)
Contains information about parameter distributions, transformations and constraints.

# Fields
$(TYPEDFIELDS)
"""
struct ParameterInfo{R<:ReConstructor,T<:TransformConstructor}
    "Contains information for flatten/unflatten parameter"
    reconstruct::R
    "Contains information for constraining and unconstraining parameter."
    transform::T
    function ParameterInfo(
        reconstruct::R, transform::T
    ) where {R<:ReConstructor,T<:TransformConstructor}
        return new{R, T}(
            reconstruct, transform
        )
    end
end
function ParameterInfo(
    flattendefault::D, constraint::C, val::B
) where {D<:FlattenDefault, C<:NamedTuple, B<:NamedTuple}
    ## Create flatten constructor
    constructor = ReConstructor(flattendefault, constraint, val)
    ## Assign transformer constraint NamedTuple
    transformer = TransformConstructor(constraint, val)
    ## Return ParameterInfo
    return ParameterInfo(
        constructor, transformer
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
    ## Create flatten constructor
    constructor = ReConstructor(flattendefault, constraint, val)
    ## Assign transformer constraint NamedTuple
    transformer = TransformConstructor(constraint, val)
    ## Return ParameterInfo
    return ParameterInfo(
        constructor, transformer
    )
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

############################################################################################
function flatten(info::ParameterInfo, x)
    return flatten(info.constructor, x)
end
function flattenAD(info::ParameterInfo, x)
    return flattenAD(info.constructor, x)
end
function unflatten(info::ParameterInfo, x)
    return unflatten(info.constructor, x)
end
function unflattenAD(info::ParameterInfo, x)
    return unflattenAD(info.constructor, x)
end

############################################################################################
#export
export ParameterInfo
