############################################################################################
#!NOTE: These are abstract super types needed if additional constraints are added to ModelWrappers.
"Compute log(abs(determinant(jacobian(`x`)))) for given transformer to unconstrained (!) domain."
function log_abs_det_jac end

"Check if (constraint, val) combination is valid. If nothing else specified, returns false (!) per default so check necessary."
function _check(_rng, constraint, val)
    return false
end

############################################################################################
#!NOTE: These are alternative functions that will default to 'constrain' and 'unconstrain' if not specified
"Inplace constrain `val` with given `constraint`, using 'val' as buffer."
function constrain!(constraint::AbstractConstraint, val)
    return constrain(constraint, val)
end

"Inplace unconstrain `val` with given `constraint`, using 'val' as buffer."
function unconstrain!(constraint::AbstractConstraint, val)
    return unconstrain(constraint, val)
end

############################################################################################
#!NOTE: Per default, flattening will not take constraint into account and just flatten based on 'x' type. Else, Implicit constraint through density by specialization -> else just flatten
function construct_flatten(
    output::Type{T},
    flattentype::F,
    unflattentype::U,
    constraint::AbstractConstraint,
    x
) where {
    T<:AbstractFloat,
    F<:FlattenTypes,
    U<:UnflattenTypes
}
    return construct_flatten(T, flattentype, unflattentype, x)
end

############################################################################################
include("params.jl")
include("random.jl")

include("constraints/constraints.jl")

include("transform.jl")

############################################################################################
# Export
export
    AbstractConstraint,
    constrain,
    unconstrain,
    log_abs_det_jac
